
# views.py (full fixed version)
from django.shortcuts import render, redirect
from .scripts import CapitalSheets_ML, CapitalSheets_Functions
from django.core.cache import cache
from .forms import ScanForm
import os
import pandas as pd
import torch
import joblib

app_dir = os.path.dirname(os.path.abspath(__file__)) 
model_path = os.path.join(app_dir, 'scripts', 'CS_Net')  
extras_path = os.path.join(app_dir, 'scripts', 'extras.joblib') 

try:
    device = torch.device('cpu')  
    input_size = 5003  
    model = CapitalSheets_ML.ImprovedMLP(input_size=input_size, hidden_size=128, num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    extras = joblib.load(extras_path)
    nn = {'model': model, **extras, 'device': device}  
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    nn = None



def generate_table_html(sheet_type, valid_tables):
    table_html = []
    indexes = range(len(valid_tables))
    for idx in indexes: 
        try:
            df = valid_tables[idx]
            df.columns = [f"{col}_{idx}" if df.columns.duplicated().any() else col for col in df.columns]
            table_html.append({
                'title': f'{sheet_type} Table {idx}',
                'html': df.to_html(index=False, classes='table table-bordered table-striped'),
                'index': idx
            })
        except IndexError:
            table_html.append({'title': f'{sheet_type}', 'html': '<p>Table not found</p>'})
    return table_html


def neural_net_scan(request, url):
    if nn is None:
        return "Model not loaded", [], [], [], [], []


    _, processed_tables = CapitalSheets_Functions.FindData(url)
    table_texts = [
        CapitalSheets_Functions.extract_table_text(df) for df in processed_tables if df is not None and not df.empty
    ]

    results = CapitalSheets_ML.predict_tokens(table_texts, nn)
    
    balance_idx, income_idx, cashflow_idx = CapitalSheets_Functions.FoundData(results)

    user_key = f"user_{request.user.id if request.user.is_authenticated else request.session.session_key}"

    cache_key_urls = f'{user_key}_urls'
    urls = cache.get(cache_key_urls, [])
    urls.append(url)
    cache.set(cache_key_urls, urls, timeout=None)


    cache.delete(f'{user_key}_nn_bal_idx')
    cache.delete(f'{user_key}_nn_inc_idx')
    cache.delete(f'{user_key}_nn_cas_idx')
    cache.delete(f'{user_key}_tables')

    current_tables = [df.to_dict('records') for df in processed_tables if df is not None and not df.empty]
    cache.set(f'{user_key}_tables', current_tables, timeout=None)

    
    cache.set(f'{user_key}_nn_bal_idx', balance_idx, timeout=None)
    cache.set(f'{user_key}_nn_inc_idx', income_idx, timeout=None)
    cache.set(f'{user_key}_nn_cas_idx', cashflow_idx, timeout=None)

    summary = "\n".join([
        f"Balance Sheets Indexes: {balance_idx}",
        f"Income Sheets Indexes: {income_idx}",
        f"Cashflow Sheets Indexes: {cashflow_idx}"
    ])


def result(request):
    if nn is None:
        return render(request, 'Backbone/result.html', {
            'form': ScanForm(),
            'error': 'Model not loaded'
        })
    
    form = ScanForm(request.POST or None)
    user_key = f"user_{request.user.id if request.user.is_authenticated else request.session.session_key}"
    
    table_dicts = cache.get(f'{user_key}_tables', [])
    valid_tables = [pd.DataFrame(d) for d in table_dicts]
    total_tables = len(valid_tables)
    scanned_urls = cache.get(f'{user_key}_urls', [])

    nn_bal_idx = cache.get(f'{user_key}_nn_bal_idx', [])
    nn_inc_idx = cache.get(f'{user_key}_nn_inc_idx', [])
    nn_cas_idx = cache.get(f'{user_key}_nn_cas_idx', [])

    balance_idx = cache.get(f'{user_key}_bal_idx', []) 
    income_idx = cache.get(f'{user_key}_inc_idx', [])
    cashflow_idx = cache.get(f'{user_key}_cas_idx', [])

    error_message = None
    
    if request.method == 'POST':
        try:
            action = request.POST.get('action')

        except:
            error_message = 'no action'
            redirect('result')

        
        if form.is_valid():
            scan_url = form.cleaned_data['url'] 

            neural_net_scan(request, scan_url)

        if action == 'reset':
            keys = [f'{user_key}_tables', f'{user_key}_urls', 
                    f'{user_key}_bal_idx', f'{user_key}_inc_idx', 
                    f'{user_key}_cas_idx', f'{user_key}_nn_bal_idx', 
                    f'{user_key}_nn_inc_idx', f'{user_key}_nn_cas_idx']
            for k in keys:
                cache.delete(k)
            return redirect('result')
            
        if action in ['export_balance', 'export_income', 'export_cashflow']:
            try:
                type = action[7:]
                fname = f'{type}_sheets.csv'
                dfs = cache.get(f'{user_key}_{type[:3]}_idx')
                pd.concat(dfs, axis=1).to_csv(fname,index=False)
            except ValueError:
                error_message = 'Can not save files, Datafranes may be empty'
                redirect('result')
            
            return redirect('result')

        if action in ['add_balance', 'add_income', 'add_cashflow']:
            try:
                idx = int(request.POST.get('index'))
                if not (0 <= idx < total_tables):
                    raise ValueError
            except:
                error_message = "Invalid index."
            else:
                lists = [
                    (balance_idx, f'{user_key}_bal_idx'),
                    (income_idx, f'{user_key}_inc_idx'),
                    (cashflow_idx, f'{user_key}_cas_idx')
                ]
                
                if action.startswith('add_'):
                   
                    try:
                        type = action[4:]
                        grabbed_data = cache.get(f'{user_key}_{type[:3]}_idx',[])
                        grabbed_data.append(valid_tables[idx].T.reset_index())
                        cache.set(f'{user_key}_{type[:3]}_idx', grabbed_data, timeout=None)

                    except ValueError:
                        error_message = 'Could not add to list'
                        redirect('result')

        if action in ['remove_bal', 'remove_inc', 'remove_cash']:
            try:
                if action == 'remove_bal':
                    idx = int(request.POST.get('bal_index'))
                    lst = balance_idx
                    key = f'{user_key}_bal_idx'

                elif action == 'remove_inc':
                    idx = int(request.POST.get('inc_index'))
                    lst = income_idx
                    key = f'{user_key}_inc_idx'

                elif action == 'remove_cash':
                    idx = int(request.POST.get('cas_index'))
                    lst = cashflow_idx
                    key = f'{user_key}_cas_idx'

                if 0 <= idx < len(lst):
                    lst.pop(idx)
                    cache.set(key, lst, timeout=None)

            except Exception as e:
                    error_message = f"Remove failed: {e}"


        return redirect('result')


    table_dicts = cache.get(f'{user_key}_tables', [])
    valid_tables = [pd.DataFrame(d) for d in table_dicts]
    total_tables = len(valid_tables)
    scanned_urls = cache.get(f'{user_key}_urls', [])

    nn_bal_idx = cache.get(f'{user_key}_nn_bal_idx', [])
    nn_inc_idx = cache.get(f'{user_key}_nn_inc_idx', [])
    nn_cas_idx = cache.get(f'{user_key}_nn_cas_idx', [])

    balance_idx = cache.get(f'{user_key}_bal_idx', [])  
    income_idx = cache.get(f'{user_key}_inc_idx', [])
    cashflow_idx = cache.get(f'{user_key}_cas_idx', [])


    selected_index = int(request.GET.get('index', 0))
    selected_table_html = "<p>No tables scanned yet.</p>"
    if total_tables > 0:
        if not (0 <= selected_index < total_tables):
            selected_index = 0
        try:
            sel_df = valid_tables[selected_index].copy()
            sel_df.columns = [f"{c}_{selected_index}" if sel_df.columns.duplicated().any() else c for c in sel_df.columns]
            selected_table_html = sel_df.T.to_html(index=True, classes='table table-bordered table-striped')
        except:
            selected_table_html = "<p>Invalid table</p>"


    table_html_balance = generate_table_html('Balance Sheet', balance_idx)
    table_html_income = generate_table_html('Income Sheet', income_idx)
    table_html_cashflow = generate_table_html('Cashflow Sheet', cashflow_idx)
    

    show_list = request.GET.get('show_list', '0') == '1'
    show_balance = request.GET.get('show_balance', '0') == '1' or show_list
    show_income = request.GET.get('show_income', '0') == '1' or show_list
    show_cashflow = request.GET.get('show_cashflow', '0') == '1' or show_list


    results_list = [
        {'sheet_type': 'Balance Sheets', 'indexes': str(nn_bal_idx)},
        {'sheet_type': 'Income Sheets', 'indexes': str(nn_inc_idx)},
        {'sheet_type': 'Cashflow Sheets', 'indexes': str(nn_cas_idx)},
    ]

    context = {
        'form': form,
        'scanned_url': scanned_urls[-1] if scanned_urls else None,
        'total_tables': total_tables,
        'selected_index': selected_index,
        'selected_table_html': selected_table_html,
        'table_html_balance': table_html_balance,
        'table_html_income': table_html_income,
        'table_html_cashflow': table_html_cashflow,
        'balance_indexes': balance_idx,
        'income_indexes': income_idx,
        'cashflow_indexes': cashflow_idx,
        'results_list': results_list,
        'error_message': error_message,
        'show_list': show_list,
        'show_balance': show_balance,
        'show_income': show_income,
        'show_cashflow': show_cashflow,
    }

    return render(request, 'Backbone/result.html', context)
