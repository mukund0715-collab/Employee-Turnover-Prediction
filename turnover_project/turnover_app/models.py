from django.db import models
from django.contrib.auth.models import User

class AnalysisResult(models.Model):
    # Link the result to the user profile
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Track the original upload name
    upload_filename = models.CharField(max_length=255)
    
    # Store the paths to the generated reports
    leavers_report_path = models.CharField(max_length=500)
    full_list_report_path = models.CharField(max_length=500)
    
    # Automatically track the date of the analysis
    analysis_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.user.username} on {self.analysis_date.strftime('%Y-%m-%d')}"
    
    class Meta:
        # Order by newest first
        ordering = ['-analysis_date']