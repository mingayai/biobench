import os
import ast
import json
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from datasets import load_dataset
import nbformat
from nbconvert import HTMLExporter

app = Flask(__name__)
app.config['CACHE_DIR'] = Path(__file__).parent / 'cache'
app.config['CACHE_DIR'].mkdir(exist_ok=True)
app.config['MAX_CAPSULE_ZIP_BYTES'] = int(os.environ.get('MAX_CAPSULE_ZIP_MB', '250')) * 1024 * 1024
app.config['MAX_CAPSULE_EXTRACTED_BYTES'] = int(os.environ.get('MAX_CAPSULE_EXTRACTED_MB', '750')) * 1024 * 1024
app.config['MIN_FREE_DISK_BYTES'] = int(os.environ.get('MIN_FREE_DISK_MB', '200')) * 1024 * 1024

DATASET_BIXBENCH = 'bixbench'
DATASET_COMPBIO = 'compbiobench'

# Columns expected on local zero-shot CSVs merged with benchmark metadata — used across
# /questions, /questions/<id>, /capsules, /compare when evaluation results are enabled.
_BIXBENCH_V15_RESULT_COLUMNS = frozenset({
    'uuid',
    'question',
    'target',
    'eval_type',
    'model',
    'correct',
    'evaluation_mode',
    'predicted',
    'grade',
})


class CapsuleTooLargeError(Exception):
    """Raised when a capsule would exceed the configured ephemeral disk budget."""


def current_dataset():
    ds = request.args.get('dataset', DATASET_BIXBENCH)
    return ds if ds in (DATASET_BIXBENCH, DATASET_COMPBIO) else DATASET_BIXBENCH


@app.context_processor
def inject_dataset():
    ds = current_dataset()

    def dataset_qs(first=False):
        if ds == DATASET_BIXBENCH:
            return ''
        sep = '?' if first else '&'
        return f'{sep}dataset={ds}'

    return dict(
        current_dataset=ds,
        dataset_qs=dataset_qs,
    )


# Global data storage
data = {
    'v1_5': None,
    'original': None,
    'v1_5_json': None,
    'original_json': None,
    'capsules': {},
    'benchmark': None,  # HuggingFace benchmark dataset
    'compbiobench': None,  # CompBioBench v1 catalog (TSV)
}

def load_data():
    """Load all CSV and JSON files on startup"""
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if hf_token:
        # `huggingface_hub` / `datasets` read HF_TOKEN; Render often sets HUGGING_FACE_HUB_TOKEN only.
        os.environ['HF_TOKEN'] = hf_token

    base_path = Path(__file__).parent / 'BixBench'

    # Load v1.5 CSVs
    v1_5_path = base_path / 'bixbench-v1.5_results' / 'zero_shot_baselines'
    v1_5_dfs = []

    for csv_file in v1_5_path.glob('*.csv'):
        filename = csv_file.stem
        df = pd.read_csv(csv_file)

        # Parse model and eval_type from filename
        if 'gpt-4o' in filename:
            df['model'] = 'gpt-4o'
        elif 'claude' in filename:
            df['model'] = 'claude-3.5-sonnet'
        else:
            print(f"Warning: skipping v1.5 CSV (cannot infer model from filename): {csv_file.name}")
            continue

        if 'mcq-refusal-False' in filename:
            df['eval_type'] = 'mcq-no-refusal'
        elif 'mcq-refusal-True' in filename:
            df['eval_type'] = 'mcq-refusal'
        elif 'openended' in filename:
            df['eval_type'] = 'openended'
        else:
            print(f"Warning: skipping v1.5 CSV (cannot infer eval type from filename): {csv_file.name}")
            continue

        v1_5_dfs.append(df)

    data['v1_5'] = pd.concat(v1_5_dfs, ignore_index=True) if v1_5_dfs else pd.DataFrame()

    # Load v1.5 JSON
    v1_5_json_path = base_path / 'bixbench-v1.5_results' / 'zero_shot_baselines.json'
    if v1_5_json_path.exists():
        with open(v1_5_json_path, 'r') as f:
            data['v1_5_json'] = json.load(f)
    else:
        data['v1_5_json'] = {}

    # Load original CSVs
    original_path = base_path / 'bixbench_results' / 'baseline_eval_data'
    original_dfs = []

    for csv_file in original_path.glob('*.csv'):
        filename = csv_file.stem
        df = pd.read_csv(csv_file)

        # Parse model and eval_type from filename
        if 'gpt-4o' in filename:
            df['model'] = 'gpt-4o'
        elif 'claude' in filename:
            df['model'] = 'claude-3.5-sonnet'
        else:
            print(f"Warning: skipping original baseline CSV (cannot infer model from filename): {csv_file.name}")
            continue

        if 'refusal_False_mcq' in filename:
            df['eval_type'] = 'mcq-no-refusal'
        elif 'refusal_True_mcq' in filename:
            df['eval_type'] = 'mcq-refusal'
        elif 'openended' in filename:
            df['eval_type'] = 'openended'
        else:
            print(f"Warning: skipping original baseline CSV (cannot infer eval type from filename): {csv_file.name}")
            continue

        original_dfs.append(df)

    data['original'] = pd.concat(original_dfs, ignore_index=True) if original_dfs else pd.DataFrame()

    # Load original JSON
    original_json_path = base_path / 'bixbench_results' / 'zero_shot_baselines.json'
    if original_json_path.exists():
        with open(original_json_path, 'r') as f:
            data['original_json'] = json.load(f)
    else:
        data['original_json'] = {}

    # Load HuggingFace benchmark dataset
    try:
        print("Loading HuggingFace BixBench dataset...")
        hf_dataset = load_dataset("futurehouse/BixBench", split="train")
        data['benchmark'] = hf_dataset.to_pandas()
        print(f"Loaded {len(data['benchmark'])} benchmark questions from HuggingFace")

        # Merge benchmark data with v1.5 results on question_id
        if data['v1_5'] is not None and not data['v1_5'].empty:
            data['v1_5'] = data['v1_5'].merge(
                data['benchmark'][['question_id', 'hypothesis', 'ideal', 'distractors', 'result', 'answer', 'paper']],
                left_on='uuid',
                right_on='question_id',
                how='left'
            )
    except Exception as e:
        print(f"Warning: Could not load HuggingFace dataset: {e}")
        data['benchmark'] = pd.DataFrame()

    # CompBioBench question catalog (TSV next to this app)
    comp_path = Path(__file__).parent / 'compbiobench.v1.tsv'
    try:
        if comp_path.exists():
            data['compbiobench'] = pd.read_csv(comp_path, sep='\t')
            print(f"Loaded {len(data['compbiobench'])} CompBioBench catalog rows from {comp_path.name}")
        else:
            print(f"Warning: CompBioBench TSV not found at {comp_path}")
            data['compbiobench'] = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not load CompBioBench TSV: {e}")
        data['compbiobench'] = pd.DataFrame()

    print(f"Loaded {len(data['v1_5'])} v1.5 results and {len(data['original'])} original results")


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=bool)
    return s.astype(str).str.strip().str.lower().isin(('true', '1', 'yes', 't'))


def _cell_bool(val) -> bool:
    return str(val).strip().lower() in ('true', '1', 'yes', 't')


def _bixbench_catalog_df() -> pd.DataFrame:
    """Return the HuggingFace BixBench catalog in the shape templates expect."""
    df = data.get('benchmark')
    if df is None or df.empty:
        return pd.DataFrame(columns=['uuid', 'question', 'target', 'capsule_id'])

    catalog = df.copy()
    if 'question_id' in catalog.columns:
        catalog['uuid'] = catalog['question_id']
    elif 'uuid' not in catalog.columns:
        catalog['uuid'] = ''

    if 'question' not in catalog.columns:
        if 'hypothesis' in catalog.columns:
            catalog['question'] = catalog['hypothesis']
        else:
            catalog['question'] = catalog['uuid']

    if 'target' not in catalog.columns:
        if 'answer' in catalog.columns:
            catalog['target'] = catalog['answer']
        elif 'result' in catalog.columns:
            catalog['target'] = catalog['result']
        else:
            catalog['target'] = ''

    catalog['capsule_id'] = catalog['uuid'].astype(str).str.extract(r'(?i)(bix-\d+)')[0]
    catalog['gpt_correct'] = None
    catalog['claude_correct'] = None
    return catalog


def _bixbench_results_available() -> bool:
    df = data.get('v1_5')
    return df is not None and not df.empty and _BIXBENCH_V15_RESULT_COLUMNS.issubset(df.columns)


def _compbiobench_dashboard():
    df = data.get('compbiobench')
    if df is None or df.empty:
        return render_template(
            'dashboard.html',
            is_compbiobench=True,
            compbiobench_empty=True,
            total_questions=0,
            domain_rows=[],
            style_rows=[],
            flag_rows=[],
            top_skills=[],
            v1_5_metrics={},
            original_metrics={},
            v1_5_chart_data={'labels': [], 'accuracy': [], 'precision': [], 'coverage': []},
            original_chart_data={'labels': [], 'accuracy': [], 'precision': [], 'coverage': []},
            eval_mode_stats=[],
        )

    total = len(df)
    if 'domain' in df.columns:
        vc_dom = df['domain'].fillna('(unknown)').value_counts().head(30)
        domain_rows = [{'domain': str(k), 'count': int(v)} for k, v in vc_dom.items()]
    else:
        domain_rows = []

    if 'question_style' in df.columns:
        vc_st = df['question_style'].fillna('(unknown)').value_counts().head(30)
        style_rows = [{'question_style': str(k), 'count': int(v)} for k, v in vc_st.items()]
    else:
        style_rows = []

    ir = _as_bool_series(df['internet_required']) if 'internet_required' in df.columns else pd.Series([False] * len(df))
    gpu = _as_bool_series(df['gpu_preferred']) if 'gpu_preferred' in df.columns else pd.Series([False] * len(df))
    flag_rows = [
        {'label': 'Internet required', 'count': int(ir.sum()), 'pct': round(100 * ir.mean(), 1) if len(ir) else 0},
        {'label': 'GPU preferred', 'count': int(gpu.sum()), 'pct': round(100 * gpu.mean(), 1) if len(gpu) else 0},
        {'label': 'Local files only (no paths)', 'count': int(((df.get('file_paths', pd.Series([''] * len(df))).fillna('').astype(str).str.strip() == '')).sum()), 'pct': 0},
    ]
    empty_fp = df['file_paths'].fillna('').astype(str).str.strip() == '' if 'file_paths' in df.columns else pd.Series(True, index=df.index)
    flag_rows[2]['count'] = int(empty_fp.sum())
    flag_rows[2]['pct'] = round(100 * empty_fp.mean(), 1) if len(df) else 0

    top_skills = []
    if 'skills_tested' in df.columns:
        exploded = (
            df['skills_tested']
            .fillna('')
            .astype(str)
            .str.split(',')
            .explode()
            .map(lambda x: x.strip())
        )
        exploded = exploded[exploded != '']
        if len(exploded):
            vc = exploded.value_counts().head(20)
            top_skills = [{'skill': k, 'count': int(v)} for k, v in vc.items()]

    return render_template(
        'dashboard.html',
        is_compbiobench=True,
        compbiobench_empty=False,
        total_questions=total,
        domain_rows=domain_rows,
        style_rows=style_rows,
        flag_rows=flag_rows,
        top_skills=top_skills,
        v1_5_metrics={},
        original_metrics={},
        v1_5_chart_data={'labels': [], 'accuracy': [], 'precision': [], 'coverage': []},
        original_chart_data={'labels': [], 'accuracy': [], 'precision': [], 'coverage': []},
        eval_mode_stats=[],
    )


def _compbiobench_questions():
    df = data.get('compbiobench')
    if df is None or df.empty:
        return render_template(
            'questions.html',
            is_compbiobench=True,
            questions=[],
            page=1,
            total_pages=1,
            total=0,
            capsules=[],
            eval_modes=[],
            eval_types=[],
            models=[],
            domains=[],
            styles=[],
            skills_options=[],
            filters={
                'search': '',
                'domain': '',
                'cb_style': '',
                'skill': '',
                'internet_required': '',
                'gpu_preferred': '',
            },
        )

    dfc = df.copy()
    search = (request.args.get('search') or '').strip()
    domain_f = request.args.get('domain') or ''
    style_f = request.args.get('cb_style') or ''
    skill_f = request.args.get('skill') or ''
    internet_f = request.args.get('internet_required') or ''
    gpu_f = request.args.get('gpu_preferred') or ''
    page = int(request.args.get('page', 1))
    per_page = 20

    if search:
        mask = pd.Series(False, index=dfc.index)
        for col in ('question_id', 'question', 'skills_tested', 'curator_name'):
            if col in dfc.columns:
                mask = mask | dfc[col].fillna('').astype(str).str.contains(search, case=False, na=False)
        dfc = dfc[mask]
    if domain_f and 'domain' in dfc.columns:
        dfc = dfc[dfc['domain'].fillna('') == domain_f]
    if style_f and 'question_style' in dfc.columns:
        dfc = dfc[dfc['question_style'].fillna('') == style_f]
    if skill_f and 'skills_tested' in dfc.columns:
        dfc = dfc[dfc['skills_tested'].fillna('').astype(str).str.contains(skill_f, case=False, na=False)]
    if internet_f in ('true', 'false') and 'internet_required' in dfc.columns:
        want = internet_f == 'true'
        dfc = dfc[_as_bool_series(dfc['internet_required']) == want]
    if gpu_f in ('true', 'false') and 'gpu_preferred' in dfc.columns:
        want = gpu_f == 'true'
        dfc = dfc[_as_bool_series(dfc['gpu_preferred']) == want]

    total = len(dfc)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    slice_df = dfc.iloc[start : start + per_page]

    questions_list = []
    for _, row in slice_df.iterrows():
        fp = row.get('file_paths', '') or ''
        parts = [p.strip() for p in str(fp).split(',') if p.strip()]
        qtext = str(row.get('question', '') or '')
        excerpt = (qtext[:200] + '…') if len(qtext) > 200 else qtext
        skills_raw = str(row.get('skills_tested', '') or '')
        skills_list = [s.strip() for s in skills_raw.split(',') if s.strip()]
        questions_list.append({
            'question_id': row.get('question_id', ''),
            'domain': row.get('domain', ''),
            'question_style': row.get('question_style', ''),
            'question_excerpt': excerpt,
            'curator_name': row.get('curator_name', ''),
            'internet_required': _cell_bool(row.get('internet_required', False)) if 'internet_required' in row.index else False,
            'gpu_preferred': _cell_bool(row.get('gpu_preferred', False)) if 'gpu_preferred' in row.index else False,
            'artifact_count': len(parts),
            'artifacts_preview': ', '.join(parts[:3]) + (' …' if len(parts) > 3 else ''),
            'skills_list': skills_list,
        })

    domains = sorted(df['domain'].dropna().unique()) if 'domain' in df.columns else []
    styles = sorted(df['question_style'].dropna().unique()) if 'question_style' in df.columns else []
    skills_set = set()
    if 'skills_tested' in df.columns:
        for s in df['skills_tested'].fillna('').astype(str):
            for part in s.split(','):
                t = part.strip()
                if t:
                    skills_set.add(t)
    skills_options = sorted(skills_set)

    return render_template(
        'questions.html',
        is_compbiobench=True,
        questions=questions_list,
        page=page,
        total_pages=total_pages,
        total=total,
        capsules=[],
        eval_modes=[],
        eval_types=[],
        models=[],
        domains=domains,
        styles=styles,
        skills_options=skills_options,
        filters={
            'search': search,
            'domain': domain_f,
            'cb_style': style_f,
            'skill': skill_f,
            'internet_required': internet_f,
            'gpu_preferred': gpu_f,
        },
    )


def _compbiobench_question_detail(qid: str):
    df = data.get('compbiobench')
    if df is None or df.empty or 'question_id' not in df.columns:
        return "Question not found", 404
    rows = df[df['question_id'].astype(str) == str(qid)]
    if len(rows) == 0:
        return "Question not found", 404
    row = rows.iloc[0]
    fp = row.get('file_paths', '') or ''
    artifacts = [p.strip() for p in str(fp).split(',') if p.strip()]
    skills_raw = str(row.get('skills_tested', '') or '')
    skills_list = [s.strip() for s in skills_raw.split(',') if s.strip()]
    internet = _cell_bool(row.get('internet_required', False)) if 'internet_required' in row.index else False
    gpu = _cell_bool(row.get('gpu_preferred', False)) if 'gpu_preferred' in row.index else False

    return render_template(
        'question.html',
        is_compbiobench=True,
        uuid=row.get('question_id', qid),
        question=str(row.get('question', '') or ''),
        target=None,
        choices=None,
        configs=[],
        capsule_id=None,
        hypothesis=None,
        ideal=None,
        distractors=None,
        result=None,
        answer=None,
        paper=None,
        cb_domain=row.get('domain'),
        cb_style=row.get('question_style'),
        cb_curator=row.get('curator_name'),
        cb_skills=skills_list,
        cb_internet=internet,
        cb_gpu=gpu,
        cb_artifacts=artifacts,
    )


@app.route('/')
def dashboard():
    """Dashboard with aggregate metrics and charts"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return _compbiobench_dashboard()

    v1_5_metrics = data.get('v1_5_json') or {}
    original_metrics = data.get('original_json') or {}

    # Prepare data for charts - v1.5
    v1_5_chart_data = {
        'labels': [],
        'accuracy': [],
        'precision': [],
        'coverage': []
    }

    for config, values in v1_5_metrics.items():
        v1_5_chart_data['labels'].append(config.replace('grader-', '').replace('-', ' '))
        v1_5_chart_data['accuracy'].append(values['accuracy'] * 100)
        v1_5_chart_data['precision'].append(values['precision'] * 100)
        v1_5_chart_data['coverage'].append(values['coverage'] * 100)

    # Prepare data for charts - original
    original_chart_data = {
        'labels': [],
        'accuracy': [],
        'precision': [],
        'coverage': []
    }

    for config, values in original_metrics.items():
        original_chart_data['labels'].append(config.replace('grader-', '').replace('-', ' '))
        original_chart_data['accuracy'].append(values['accuracy'] * 100)
        original_chart_data['precision'].append(values['precision'] * 100)
        original_chart_data['coverage'].append(values['coverage'] * 100)

    # Compute per-evaluation_mode breakdown from v1.5 data
    eval_mode_stats = []
    if (
        data['v1_5'] is not None
        and not data['v1_5'].empty
        and {'eval_type', 'evaluation_mode', 'correct', 'model'}.issubset(data['v1_5'].columns)
    ):
        df_v1_5 = data['v1_5']
        # Filter to mcq-no-refusal for consistent comparison
        df_mcq = df_v1_5[df_v1_5['eval_type'] == 'mcq-no-refusal']
        
        for eval_mode in sorted(df_mcq['evaluation_mode'].dropna().unique()):
            mode_data = df_mcq[df_mcq['evaluation_mode'] == eval_mode]
            
            # Overall accuracy
            overall_correct = mode_data['correct'].sum()
            overall_total = len(mode_data)
            overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
            
            # GPT-4o accuracy
            gpt_data = mode_data[mode_data['model'] == 'gpt-4o']
            gpt_correct = gpt_data['correct'].sum()
            gpt_total = len(gpt_data)
            gpt_accuracy = (gpt_correct / gpt_total * 100) if gpt_total > 0 else 0
            
            # Claude accuracy
            claude_data = mode_data[mode_data['model'] == 'claude-3.5-sonnet']
            claude_correct = claude_data['correct'].sum()
            claude_total = len(claude_data)
            claude_accuracy = (claude_correct / claude_total * 100) if claude_total > 0 else 0
            
            eval_mode_stats.append({
                'mode': eval_mode,
                'overall_accuracy': overall_accuracy,
                'gpt_accuracy': gpt_accuracy,
                'claude_accuracy': claude_accuracy,
                'total_questions': overall_total // 2  # Divide by 2 since we have both models
            })

    return render_template(
        'dashboard.html',
        is_compbiobench=False,
        bixbench_results_missing=not _bixbench_results_available(),
        v1_5_metrics=v1_5_metrics,
        original_metrics=original_metrics,
        v1_5_chart_data=v1_5_chart_data,
        original_chart_data=original_chart_data,
        eval_mode_stats=eval_mode_stats,
    )


@app.route('/questions')
def questions():
    """Filterable question list"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return _compbiobench_questions()

    if not _bixbench_results_available():
        return _bixbench_catalog_questions()

    df = data['v1_5'].copy()

    # Apply filters
    capsule = request.args.get('capsule')
    eval_mode = request.args.get('eval_mode')
    eval_type = request.args.get('eval_type')
    model = request.args.get('model')
    correct = request.args.get('correct')
    page = int(request.args.get('page', 1))
    per_page = 20

    if capsule:
        df = df[df['uuid'].str.contains(capsule, case=False, na=False)]
    if eval_mode and 'evaluation_mode' in df.columns:
        df = df[df['evaluation_mode'] == eval_mode]
    if eval_type:
        df = df[df['eval_type'] == eval_type]
    if model:
        df = df[df['model'] == model]
    if correct:
        df = df[df['correct'] == (correct.lower() == 'true')]

    # Get unique questions (aggregate across configs)
    # If no specific model/eval_type is selected, show mcq-no-refusal results for consistency
    if not model and not eval_type:
        df_display = df[df['eval_type'] == 'mcq-no-refusal'].copy()
        if len(df_display) == 0:
            df_display = df.copy()
    else:
        df_display = df.copy()

    unique_questions = df_display.drop_duplicates(subset=['uuid', 'question'])
    unique_questions['capsule_id'] = unique_questions['uuid'].str.extract(r'(?i)(bix-\d+)')[0]
    
    # Add per-model correctness for the list view
    # Get GPT-4o and Claude results for each question (using the same eval_type as df_display)
    questions_list = []
    for _, q in unique_questions.iterrows():
        q_uuid = q['uuid']
        q_eval_type = q.get('eval_type', 'mcq-no-refusal')
        
        # Get GPT-4o result for this question
        gpt_results = df_display[(df_display['uuid'] == q_uuid) & (df_display['model'] == 'gpt-4o')]
        gpt_correct = gpt_results.iloc[0]['correct'] if len(gpt_results) > 0 else None
        
        # Get Claude result for this question
        claude_results = df_display[(df_display['uuid'] == q_uuid) & (df_display['model'] == 'claude-3.5-sonnet')]
        claude_correct = claude_results.iloc[0]['correct'] if len(claude_results) > 0 else None
        
        q_dict = q.to_dict()
        q_dict['gpt_correct'] = gpt_correct
        q_dict['claude_correct'] = claude_correct
        questions_list.append(q_dict)

    # Pagination
    total = len(questions_list)
    start = (page - 1) * per_page
    end = start + per_page
    questions_page = questions_list[start:end]

    # Get dropdowns for filter form
    capsules = sorted(df['uuid'].str.split('-q').str[0].unique())
    eval_modes = (
        sorted(df['evaluation_mode'].dropna().unique()) if 'evaluation_mode' in df.columns else []
    )
    eval_types = sorted(df['eval_type'].unique()) if 'eval_type' in df.columns else []
    models = sorted(df['model'].unique()) if 'model' in df.columns else []

    return render_template(
        'questions.html',
        is_compbiobench=False,
        questions=questions_page,
        page=page,
        total_pages=(total + per_page - 1) // per_page,
        total=total,
        capsules=capsules,
        eval_modes=eval_modes,
        eval_types=eval_types,
        models=models,
        domains=[],
        styles=[],
        skills_options=[],
        filters={
            'capsule': capsule,
            'eval_mode': eval_mode,
            'eval_type': eval_type,
            'model': model,
            'correct': correct,
        },
    )


def _bixbench_catalog_questions():
    """Question list fallback when local evaluation result CSVs are not deployed."""
    df = _bixbench_catalog_df()
    capsule = request.args.get('capsule')
    page = int(request.args.get('page', 1))
    per_page = 20

    if capsule and not df.empty:
        df = df[df['uuid'].astype(str).str.contains(capsule, case=False, na=False)]

    questions_list = df.drop_duplicates(subset=['uuid']).to_dict('records') if not df.empty else []
    total = len(questions_list)
    start = (page - 1) * per_page
    end = start + per_page

    all_catalog = _bixbench_catalog_df()
    capsules = sorted(all_catalog['capsule_id'].dropna().unique()) if not all_catalog.empty else []

    return render_template(
        'questions.html',
        is_compbiobench=False,
        questions=questions_list[start:end],
        page=page,
        total_pages=max(1, (total + per_page - 1) // per_page),
        total=total,
        capsules=capsules,
        eval_modes=[],
        eval_types=[],
        models=[],
        domains=[],
        styles=[],
        skills_options=[],
        filters={
            'capsule': capsule,
            'eval_mode': None,
            'eval_type': None,
            'model': None,
            'correct': None,
        },
    )


@app.route('/questions/<qid>')
def question_detail(qid):
    """Single question detail view"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return _compbiobench_question_detail(qid)
    if not _bixbench_results_available():
        return _bixbench_catalog_question_detail(qid)

    # Get question from v1.5 data
    v1_5_rows = data['v1_5'][data['v1_5']['uuid'] == qid]

    # Get question from original data (for choices if available)
    original_rows = data['original'][data['original']['uuid'] == qid]

    if len(v1_5_rows) == 0:
        return "Question not found", 404

    question_text = v1_5_rows.iloc[0]['question']
    target = v1_5_rows.iloc[0]['target']
    uuid = v1_5_rows.iloc[0]['uuid']

    # Extract capsule_id from uuid
    import re
    capsule_match = re.search(r'(?i)(bix-\d+)', uuid)
    capsule_id = capsule_match.group(1) if capsule_match else None

    # Get benchmark metadata (hypothesis, ideal, distractors)
    hypothesis = v1_5_rows.iloc[0].get('hypothesis') if 'hypothesis' in v1_5_rows.columns else None
    ideal = v1_5_rows.iloc[0].get('ideal') if 'ideal' in v1_5_rows.columns else None

    # Convert distractors to list if available
    distractors = None
    if 'distractors' in v1_5_rows.columns:
        distractors_raw = v1_5_rows.iloc[0].get('distractors')
        try:
            if pd.notna(distractors_raw):
                distractors = list(distractors_raw) if hasattr(distractors_raw, '__iter__') and not isinstance(distractors_raw, str) else None
        except (ValueError, TypeError):
            distractors = None

    result = v1_5_rows.iloc[0].get('result') if 'result' in v1_5_rows.columns else None

    # Convert answer to Python bool
    answer = None
    if 'answer' in v1_5_rows.columns:
        answer_raw = v1_5_rows.iloc[0].get('answer')
        try:
            if pd.notna(answer_raw):
                answer = bool(answer_raw)
        except (ValueError, TypeError):
            answer = None

    paper = v1_5_rows.iloc[0].get('paper') if 'paper' in v1_5_rows.columns else None

    # Get choices if available
    choices = None
    if len(original_rows) > 0 and 'choices' in original_rows.columns:
        choices_raw = original_rows.iloc[0]['choices']
        if pd.notna(choices_raw):
            try:
                choices = ast.literal_eval(choices_raw) if isinstance(choices_raw, str) else choices_raw
            except (ValueError, SyntaxError, TypeError):
                choices = None

    # Get all configurations for this question
    cfg_cols = ['model', 'eval_type', 'predicted', 'grade', 'correct', 'evaluation_mode']
    if 'sure' in v1_5_rows.columns:
        cfg_cols.append('sure')
    configs = v1_5_rows[cfg_cols].to_dict('records')
    if 'sure' not in cfg_cols:
        for row in configs:
            row['sure'] = False

    return render_template(
        'question.html',
        is_compbiobench=False,
        uuid=uuid,
        question=question_text,
        target=target,
        choices=choices,
        configs=configs,
        capsule_id=capsule_id,
        hypothesis=hypothesis,
        ideal=ideal,
        distractors=distractors,
        result=result,
        answer=answer,
        paper=paper,
    )


def _bixbench_catalog_question_detail(qid):
    catalog = _bixbench_catalog_df()
    rows = catalog[catalog['uuid'] == qid] if not catalog.empty else pd.DataFrame()
    if len(rows) == 0:
        return "Question not found", 404

    row = rows.iloc[0]
    distractors = row.get('distractors') if 'distractors' in rows.columns else None
    if hasattr(distractors, '__iter__') and not isinstance(distractors, str):
        distractors = list(distractors)
    else:
        distractors = None

    return render_template(
        'question.html',
        is_compbiobench=False,
        uuid=row.get('uuid'),
        question=row.get('question'),
        target=row.get('target'),
        choices=None,
        configs=[],
        capsule_id=row.get('capsule_id'),
        hypothesis=row.get('hypothesis') if 'hypothesis' in rows.columns else None,
        ideal=row.get('ideal') if 'ideal' in rows.columns else None,
        distractors=distractors,
        result=row.get('result') if 'result' in rows.columns else None,
        answer=row.get('answer') if 'answer' in rows.columns else None,
        paper=row.get('paper') if 'paper' in rows.columns else None,
    )


@app.route('/capsules')
def capsules():
    """Capsules overview grouped by UUID"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return redirect(url_for('dashboard', dataset=DATASET_COMPBIO))

    if not _bixbench_results_available():
        return _bixbench_catalog_capsules()

    df = data['v1_5'].copy()
    
    # Filter to mcq-no-refusal for consistent comparison
    df = df[df['eval_type'] == 'mcq-no-refusal']

    # Extract capsule ID from UUID (e.g., bix-1-q1 -> bix-1)
    df['capsule_id'] = df['uuid'].str.extract(r'(?i)(bix-\d+)')[0]

    # Group by capsule
    capsule_stats = []
    for capsule_id in sorted(df['capsule_id'].dropna().unique()):
        capsule_data = df[df['capsule_id'] == capsule_id]

        # Calculate per-model accuracy (now scoped to mcq-no-refusal)
        gpt_accuracy = capsule_data[capsule_data['model'] == 'gpt-4o']['correct'].mean()
        claude_accuracy = capsule_data[capsule_data['model'] == 'claude-3.5-sonnet']['correct'].mean()

        capsule_stats.append({
            'id': capsule_id,
            'total_questions': len(capsule_data['uuid'].unique()),
            'gpt_accuracy': gpt_accuracy * 100 if pd.notna(gpt_accuracy) else 0,
            'claude_accuracy': claude_accuracy * 100 if pd.notna(claude_accuracy) else 0,
            'overall_accuracy': capsule_data['correct'].mean() * 100
        })

    # Sort by overall accuracy (ascending = hardest first)
    capsule_stats.sort(key=lambda x: x['overall_accuracy'])

    return render_template('capsules.html', capsules=capsule_stats)


def _bixbench_catalog_capsules():
    catalog = _bixbench_catalog_df()
    capsule_stats = []
    if not catalog.empty:
        for capsule_id in sorted(catalog['capsule_id'].dropna().unique()):
            capsule_data = catalog[catalog['capsule_id'] == capsule_id]
            capsule_stats.append({
                'id': capsule_id,
                'total_questions': len(capsule_data['uuid'].unique()),
                'gpt_accuracy': 0,
                'claude_accuracy': 0,
                'overall_accuracy': 0,
            })

    return render_template('capsules.html', capsules=capsule_stats, has_eval_results=False)


@app.route('/capsules/<capsule_id>')
def capsule_detail(capsule_id):
    """Single capsule detail view"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return redirect(url_for('dashboard', dataset=DATASET_COMPBIO))

    if not _bixbench_results_available():
        return _bixbench_catalog_capsule_detail(capsule_id)

    df = data['v1_5'].copy()
    
    # Filter to mcq-no-refusal for consistent comparison
    df_mcq = df[df['eval_type'] == 'mcq-no-refusal']
    df_mcq['capsule_id'] = df_mcq['uuid'].str.extract(r'(?i)(bix-\d+)')[0]

    capsule_data = df_mcq[df_mcq['capsule_id'] == capsule_id]

    if len(capsule_data) == 0:
        return "Capsule not found", 404

    # Get unique questions
    questions = capsule_data.drop_duplicates(subset=['uuid'])

    # Calculate stats (now scoped to mcq-no-refusal)
    stats = {
        'total_questions': len(questions),
        'gpt_accuracy': capsule_data[capsule_data['model'] == 'gpt-4o']['correct'].mean() * 100,
        'claude_accuracy': capsule_data[capsule_data['model'] == 'claude-3.5-sonnet']['correct'].mean() * 100,
        'overall_accuracy': capsule_data['correct'].mean() * 100
    }

    # Check if data is loaded
    data_loaded = capsule_id in data['capsules']
    capsule_info = data['capsules'].get(capsule_id, {})
    file_list = capsule_info.get('files', []) if data_loaded else []
    has_notebook = capsule_info.get('notebook_path') is not None if data_loaded else False

    return render_template('capsule.html',
                         capsule_id=capsule_id,
                         stats=stats,
                         questions=questions.to_dict('records'),
                         data_loaded=data_loaded,
                         file_list=file_list,
                         has_notebook=has_notebook)


def _bixbench_catalog_capsule_detail(capsule_id):
    catalog = _bixbench_catalog_df()
    capsule_data = catalog[catalog['capsule_id'] == capsule_id] if not catalog.empty else pd.DataFrame()

    if len(capsule_data) == 0:
        return "Capsule not found", 404

    questions = capsule_data.drop_duplicates(subset=['uuid'])
    stats = {
        'total_questions': len(questions),
        'gpt_accuracy': 0,
        'claude_accuracy': 0,
        'overall_accuracy': 0,
    }

    data_loaded = capsule_id in data['capsules']
    capsule_info = data['capsules'].get(capsule_id, {})
    file_list = capsule_info.get('files', []) if data_loaded else []
    has_notebook = capsule_info.get('notebook_path') is not None if data_loaded else False

    return render_template('capsule.html',
                         capsule_id=capsule_id,
                         stats=stats,
                         questions=questions.to_dict('records'),
                         data_loaded=data_loaded,
                         file_list=file_list,
                         has_notebook=has_notebook,
                         has_eval_results=False)


def _format_mb(byte_count: Optional[int]) -> str:
    if byte_count is None:
        return 'unknown size'
    return f'{byte_count / 1024 / 1024:.1f} MB'


def _disk_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def _enough_disk_for(required_bytes: int, path: Path) -> bool:
    free_after_write = _disk_free_bytes(path) - required_bytes
    return free_after_write >= app.config['MIN_FREE_DISK_BYTES']


def _capsule_too_large_response(error: Exception):
    return jsonify({
        'success': False,
        'error': str(error),
        'too_large': True,
    }), 413


def _get_hf_file_size(repo_id: str, filename: str) -> Optional[int]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type='dataset')
    metadata = get_hf_file_metadata(url)
    return metadata.size


def _assert_download_fits(filename: str, zip_size: Optional[int]) -> None:
    max_zip = app.config['MAX_CAPSULE_ZIP_BYTES']
    if zip_size is not None and zip_size > max_zip:
        raise CapsuleTooLargeError(
            f'Capsule data file "{filename}" is {_format_mb(zip_size)}, which is larger than '
            f'the configured free-tier ZIP budget of {_format_mb(max_zip)}.'
        )

    if zip_size is not None and not _enough_disk_for(zip_size, app.config['CACHE_DIR']):
        raise CapsuleTooLargeError(
            f'Capsule data file "{filename}" needs {_format_mb(zip_size)} to download, but this '
            f'instance does not have enough free ephemeral disk after reserving '
            f'{_format_mb(app.config["MIN_FREE_DISK_BYTES"])}.'
        )


def _zip_uncompressed_size(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return sum(info.file_size for info in zip_ref.infolist() if not info.is_dir())


def _assert_extraction_fits(zip_path: Path) -> None:
    expanded_size = _zip_uncompressed_size(zip_path)
    max_extracted = app.config['MAX_CAPSULE_EXTRACTED_BYTES']

    if expanded_size > max_extracted:
        raise CapsuleTooLargeError(
            f'Capsule expands to {_format_mb(expanded_size)}, which is larger than the configured '
            f'free-tier extraction budget of {_format_mb(max_extracted)}.'
        )

    if not _enough_disk_for(expanded_size, app.config['CACHE_DIR']):
        raise CapsuleTooLargeError(
            f'Capsule needs about {_format_mb(expanded_size)} to extract, but this instance does '
            f'not have enough free ephemeral disk after reserving '
            f'{_format_mb(app.config["MIN_FREE_DISK_BYTES"])}.'
        )


def _extract_and_process_capsule(zip_path: Path, extract_path: Path) -> Tuple[Optional[Path], list]:
    """
    Extract and process capsule files, preserving notebooks.
    Returns (notebook_path, file_list)
    """
    # Unzip the archive
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Find the Data folder (name contains "Data")
    data_folder = next(
        (p for p in extract_path.rglob("*") if p.is_dir() and "Data" in p.name),
        None
    )
    
    if data_folder:
        # Move contents of Data folder to parent directory
        for item in data_folder.iterdir():
            shutil.move(str(item), str(extract_path / item.name))
        # Remove the now-empty Data folder
        shutil.rmtree(data_folder)
    
    # Find the Notebook folder (name contains "Notebook") and locate notebook file
    notebook_path = None
    notebook_folder = next(
        (p for p in extract_path.rglob("*") if p.is_dir() and "Notebook" in p.name),
        None
    )
    
    if notebook_folder:
        # Find the executed notebook
        notebook_file = next(
            (p for p in notebook_folder.rglob("*.ipynb") if "_executed" in p.name),
            next(notebook_folder.rglob("*.ipynb"), None)  # Fallback to any notebook
        )
        if notebook_file:
            notebook_path = notebook_file
    
    # Build file listing (data files only, not notebook)
    files = []
    for item in extract_path.rglob('*'):
        if item.is_file() and not item.suffix == '.ipynb':
            # Skip files in notebook folders
            if notebook_folder and notebook_folder in item.parents:
                continue
            rel_path = item.relative_to(extract_path)
            files.append({
                'path': str(rel_path),
                'size': item.stat().st_size,
                'name': item.name
            })
    
    return notebook_path, files

@app.route('/api/capsule/<capsule_id>/load')
def load_capsule_data(capsule_id):
    """Download and extract capsule data from HuggingFace"""
    try:
        # First, we need to find the actual capsule UUID from the benchmark dataset
        # The capsule_id parameter is like "bix-9", but HuggingFace files are named by UUID
        capsule_uuid = None
        
        if data['benchmark'] is not None and not data['benchmark'].empty:
            # Find any question from this capsule to get the capsule_uuid
            capsule_rows = data['benchmark'][data['benchmark']['question_id'].str.startswith(capsule_id + '-')]
            if len(capsule_rows) > 0 and 'capsule_uuid' in data['benchmark'].columns:
                capsule_uuid = capsule_rows.iloc[0]['capsule_uuid']
        
        if not capsule_uuid:
            return jsonify({
                'success': False,
                'error': f'Could not find capsule UUID for {capsule_id} in the dataset. The benchmark metadata may not include capsule information.',
                'not_found': True
            }), 404

        # Download ZIP from HuggingFace using the actual UUID filename
        zip_filename = f"CapsuleFolder-{capsule_uuid}.zip"
        repo_id = "futurehouse/bixbench"
        try:
            zip_size = _get_hf_file_size(repo_id, zip_filename)
            _assert_download_fits(zip_filename, zip_size)
            zip_path = hf_hub_download(
                repo_id=repo_id,
                filename=zip_filename,
                repo_type="dataset",
                cache_dir=app.config['CACHE_DIR'] / '_hf'
            )
        except Exception as e:
            # If file not found on HuggingFace, return a helpful message
            if "404" in str(e) or "Entry Not Found" in str(e):
                return jsonify({
                    'success': False,
                    'error': f'Capsule data file "{zip_filename}" not available in the HuggingFace repository. This capsule may not have uploaded data files.',
                    'not_found': True
                }), 404
            if isinstance(e, CapsuleTooLargeError):
                return _capsule_too_large_response(e)
            raise

        # Extract to cache directory
        extract_path = app.config['CACHE_DIR'] / capsule_id
        tmp_extract_path = app.config['CACHE_DIR'] / f'.extracting-{capsule_id}'

        try:
            _assert_extraction_fits(Path(zip_path))
            if tmp_extract_path.exists():
                shutil.rmtree(tmp_extract_path)
            tmp_extract_path.mkdir(parents=True)

            # Extract and process capsule, preserving notebooks
            notebook_path, files = _extract_and_process_capsule(Path(zip_path), tmp_extract_path)

            if extract_path.exists():
                shutil.rmtree(extract_path)
            shutil.move(str(tmp_extract_path), str(extract_path))
            if notebook_path:
                notebook_path = extract_path / notebook_path.relative_to(tmp_extract_path)
        except CapsuleTooLargeError as e:
            if tmp_extract_path.exists():
                shutil.rmtree(tmp_extract_path)
            return _capsule_too_large_response(e)
        except Exception:
            if tmp_extract_path.exists():
                shutil.rmtree(tmp_extract_path)
            data['capsules'].pop(capsule_id, None)
            raise

        # Store in memory
        data['capsules'][capsule_id] = {
            'path': extract_path,
            'files': files,
            'notebook_path': notebook_path,
            'zip_path': Path(zip_path)
        }

        return jsonify({
            'success': True,
            'files': files,
            'has_notebook': notebook_path is not None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/capsule/<capsule_id>/file/<path:filepath>')
def get_capsule_file(capsule_id, filepath):
    """Get preview of a capsule data file"""
    if capsule_id not in data['capsules']:
        return jsonify({'error': 'Capsule not loaded'}), 404

    capsule_path = data['capsules'][capsule_id]['path'].resolve()
    file_path = (capsule_path / filepath).resolve()

    try:
        file_path.relative_to(capsule_path)
    except ValueError:
        return jsonify({'error': 'Invalid file path'}), 400

    if not file_path.exists() or not file_path.is_file():
        return jsonify({'error': 'File not found'}), 404

    # Read first 50 lines for preview
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [f.readline() for _ in range(50)]

        return jsonify({
            'success': True,
            'preview': ''.join(lines),
            'path': filepath
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/capsule/<capsule_id>/notebook')
def get_capsule_notebook(capsule_id):
    """Render capsule notebook as HTML"""
    if capsule_id not in data['capsules']:
        return jsonify({'error': 'Capsule not loaded'}), 404
    
    capsule_info = data['capsules'][capsule_id]
    notebook_path = capsule_info.get('notebook_path')
    
    if not notebook_path or not notebook_path.exists():
        return jsonify({'error': 'Notebook not found for this capsule'}), 404
    
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Convert to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        return jsonify({
            'success': True,
            'html': body,
            'notebook_name': notebook_path.name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error rendering notebook: {str(e)}'
        }), 500

@app.route('/compare')
def compare():
    """Model comparison - agreement/disagreement"""
    ds = current_dataset()
    if ds == DATASET_COMPBIO:
        return redirect(url_for('dashboard', dataset=DATASET_COMPBIO))

    if not _bixbench_results_available():
        return render_template('compare.html',
                             comparisons=[],
                             counts={'both_correct': 0, 'both_wrong': 0, 'gpt_only': 0, 'claude_only': 0},
                             percentages={'both_correct': 0, 'both_wrong': 0, 'gpt_only': 0, 'claude_only': 0},
                             total=0,
                             total_filtered=0,
                             page=1,
                             total_pages=1,
                             filter_cat=request.args.get('category'))

    df = data['v1_5'].copy()

    # Pivot to get GPT-4o vs Claude results per question
    comparison_data = []

    for uuid in df['uuid'].unique():
        question_data = df[df['uuid'] == uuid]

        gpt_results = question_data[question_data['model'] == 'gpt-4o']
        claude_results = question_data[question_data['model'] == 'claude-3.5-sonnet']

        if len(gpt_results) > 0 and len(claude_results) > 0:
            # Use MCQ no-refusal as primary comparison
            gpt_mcq = gpt_results[gpt_results['eval_type'] == 'mcq-no-refusal']
            claude_mcq = claude_results[claude_results['eval_type'] == 'mcq-no-refusal']

            if len(gpt_mcq) > 0 and len(claude_mcq) > 0:
                gpt_correct = gpt_mcq.iloc[0]['correct']
                claude_correct = claude_mcq.iloc[0]['correct']

                if gpt_correct and claude_correct:
                    category = 'both_correct'
                elif not gpt_correct and not claude_correct:
                    category = 'both_wrong'
                elif gpt_correct:
                    category = 'gpt_only'
                else:
                    category = 'claude_only'

                comparison_data.append({
                    'uuid': uuid,
                    'question': question_data.iloc[0]['question'][:100] + '...',
                    'category': category,
                    'gpt_predicted': gpt_mcq.iloc[0]['predicted'],
                    'claude_predicted': claude_mcq.iloc[0]['predicted'],
                    'target': gpt_mcq.iloc[0]['target']
                })

    # Calculate counts
    category_counts = {
        'both_correct': sum(1 for x in comparison_data if x['category'] == 'both_correct'),
        'both_wrong': sum(1 for x in comparison_data if x['category'] == 'both_wrong'),
        'gpt_only': sum(1 for x in comparison_data if x['category'] == 'gpt_only'),
        'claude_only': sum(1 for x in comparison_data if x['category'] == 'claude_only')
    }

    total = len(comparison_data)
    percentages = {k: (v / total * 100 if total > 0 else 0) for k, v in category_counts.items()}

    # Filter by category if requested
    filter_cat = request.args.get('category')
    if filter_cat:
        comparison_data = [x for x in comparison_data if x['category'] == filter_cat]

    # Add pagination
    page = int(request.args.get('page', 1))
    per_page = 20
    total_filtered = len(comparison_data)
    start = (page - 1) * per_page
    end = start + per_page
    comparisons_page = comparison_data[start:end]

    return render_template('compare.html',
                         comparisons=comparisons_page,
                         counts=category_counts,
                         percentages=percentages,
                         total=total,
                         total_filtered=total_filtered,
                         page=page,
                         total_pages=(total_filtered + per_page - 1) // per_page,
                         filter_cat=filter_cat)

if __name__ == '__main__':
    load_data()
    app.run(
        debug=os.environ.get('FLASK_DEBUG') == '1',
        host='0.0.0.0',
        port=int(os.environ.get('PORT', '5001')),
    )
