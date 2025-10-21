import os
import pandas as pd
import tempfile
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, FileResponse, Http404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

# Import your core analysis script
# NOTE: The import path needs to be correct relative to your app structure
from .main_script import run_turnover_analysis 
from .models import AnalysisResult

# ====================================================================
# A. Authentication Views (Using Django's built-in forms)
# ====================================================================

def register_view(request):
    """Handles user registration."""
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Log the user in immediately
            return redirect('profile')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    """Handles user login."""
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('profile')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    """Handles user logout."""
    logout(request)
    return redirect('home')

# ====================================================================
# B. Main Application Views
# ====================================================================

def home(request):
    """Home page with project information."""
    return render(request, 'home.html')

@login_required
def profile(request):
    """User profile page showing analysis history and upload form."""
    # Retrieve all analysis results for the currently logged-in user
    results = AnalysisResult.objects.filter(user=request.user)
    
    context = {
        'results': results,
    }
    return render(request, 'profile.html', context)

@login_required
def upload_file(request):
    """Handles the file upload and runs the ML analysis pipeline."""
    if request.method == 'POST' and request.FILES.get('data_file'):
        uploaded_file = request.FILES['data_file']
        
        # 1. Save the file temporarily to run the script
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Ensure the file is a CSV, converting from Excel if necessary
        try:
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
                # Save the Excel data as a temporary CSV
                temp_file_path = temp_file_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
                df.to_csv(temp_file_path, index=False)
            else:
                # Save the uploaded CSV content directly
                with open(temp_file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

            # 2. Run the analysis (from main_script.py)
            base_name = os.path.splitext(uploaded_file.name)[0]
            leavers_report_path, full_list_report_path = run_turnover_analysis(
                temp_file_path, base_name
            )

            # 3. Save the result paths to the database
            AnalysisResult.objects.create(
                user=request.user,
                upload_filename=uploaded_file.name,
                leavers_report_path=leavers_report_path,
                full_list_report_path=full_list_report_path,
            )

        except Exception as e:
            # Handle potential errors (e.g., malformed data, script crash)
            print(f"Analysis failed: {e}")
            # Optional: Add error message to Django messages framework
            return redirect('profile') # Redirect back to profile on failure

        finally:
            # Clean up the initial uploaded file from temp, 
            # but keep the report files as they are needed for download
            if os.path.exists(temp_file_path):
                # Only remove the file if it's the temporary initial upload
                # The run_turnover_analysis places the outputs in temp_dir as well
                pass # NOTE: Your current main_script saves outputs to temp, so no cleanup needed here, but in a production system, you'd save outputs to MEDIA_ROOT.

        return redirect('profile')

    return redirect('profile') # If accessed via GET

@login_required
def download_file(request, result_id, file_type, mode):
    """
    Serves the requested report file.
    
    Args:
        mode (str): 'view' to open in browser, 'download' to force download.
    """
    
    result = get_object_or_404(AnalysisResult, id=result_id, user=request.user)
    
    # 1. Determine which file path and name to use
    if file_type == 'leavers':
        file_path = result.leavers_report_path
        file_name = f"{result.upload_filename.split('.')[0]}_priority_leavers_report.csv"
    elif file_type == 'full':
        file_path = result.full_list_report_path
        file_name = f"{result.upload_filename.split('.')[0]}_priority_full_list_report.csv"
    else:
        raise Http404("Invalid file type.")

    # 2. Check if the file exists on the server
    if not os.path.exists(file_path):
        raise Http404("Report file not found.")

    # 3. Configure the HTTP Response based on the mode
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')

    if mode == 'download':
        # Force the browser to download the file
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = f'attachment; filename="{file_name}"'
    elif mode == 'view':
        # Suggest the browser display the file content (especially good for CSVs)
        # Note: This uses 'inline', but the browser ultimately decides (usually works for CSV)
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = f'inline; filename="{file_name}"'
        
    return response

# Update the turnover_app/urls.py to map the new view functions:
# from django.urls import path
# from . import views
# 
# urlpatterns = [
#     path('', views.home, name='home'),
#     path('register/', views.register_view, name='register'), # NEW
#     path('login/', views.login_view, name='login'),         # NEW
#     path('logout/', views.logout_view, name='logout'),       # NEW
#     path('profile/', views.profile, name='profile'),
#     path('upload/', views.upload_file, name='upload_file'),
#     path('download/<int:result_id>/<str:file_type>/', views.download_file, name='download_file'),
# ]