import os
import pandas as pd
import tempfile
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse, FileResponse, Http404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.core.files.base import ContentFile

# Import your core analysis script
# NOTE: The import path needs to be correct relative to your app structure
from .main_script import run_turnover_analysis, run_visualization
from .models import AnalysisResult, ResultImage

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
    
    # Get results AND pre-fetch all related images in one go
    results = AnalysisResult.objects.filter(
        user=request.user
    ).prefetch_related(
        'visualizations'  # The name of your related model
    ).order_by('-analysis_date') # Good to order them, newest first
    
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
            Analysis_object = AnalysisResult.objects.create(
                user=request.user,
                upload_filename=uploaded_file.name,
                leavers_report_path=leavers_report_path,
                full_list_report_path=full_list_report_path,
            )
            
            temp_path_list = run_visualization(temp_file_path, full_list_report_path, leavers_report_path)

# 3. Loop over the paths and create the MANY ResultImage objects
            for temp_path in temp_path_list:
                
                # Open the image file from its temporary path
                with open(temp_path, 'rb') as f:
                    print(temp_path)
                    # We need to wrap the file content for Django
                    # We also get the filename (e.g., 'viz_1.png') from the path
                    image_file = temp_path

                    # Create the ResultImage, linking it to the 'analysis_result'
                    ResultImage.objects.create(
                        result=Analysis_object,  # <-- The link!
                        image=image_file,
                        alt_text="Visualization image"
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
    Serves the requested report file OR redirects to the dashboard.
    """
    result = get_object_or_404(AnalysisResult, id=result_id, user=request.user)

    # --- THIS IS THE FIX ---
    # If the user clicks "View", redirect to the new dashboard URL
    if mode == 'view':
        return redirect('dashboard', result_id=result.id)
    # --- END OF FIX ---

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

    # 3. Configure the HTTP Response (only 'download' mode will reach here)
    response = FileResponse(open(file_path, 'rb'), content_type='text/csv')
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = f'attachment; filename="{file_name}"'
        
    return response

@login_required
def dashboard_view(request, result_id):
    """
    Displays the visualization dashboard for a SINGLE analysis result.
    """
    # 1. Get the specific result for this user
    result = get_object_or_404(AnalysisResult, id=result_id, user=request.user)
    
    # 2. Get all visualization images linked to this result
    visualizations = result.visualizations.all()
    
    context = {
        'result': result,
        'visualizations': visualizations,
    }
    return render(request, 'dashboard.html', context)
