from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_view, name='register'),
     path('login/', views.login_view, name='login'),         
     path('logout/', views.logout_view, name='logout'), 
    path('profile/', views.profile, name='profile'),
    path('upload/', views.upload_file, name='upload_file'),
    path('download/<int:result_id>/<str:file_type>/', views.download_file, name='download_file'),
    # Django Auth URLs (for login, logout, registration)
    path('accounts/login/', views.login_view, name='login'), # Define a custom or use built-in
    path('accounts/logout/', views.logout_view, name='logout'), # Define a custom or use built-in
    path('download/<int:result_id>/<str:file_type>/<str:mode>/', views.download_file, name='download_file'),
    path('dashboard/<int:result_id>/', views.dashboard_view, name='dashboard')
]
