from django.db import models
from django.contrib.auth.models import User

class AnalysisResult(models.Model):
    # Link the result to the user profile
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Track the original upload name
    upload_filename = models.CharField(max_length=255)
    leavers_report_path = models.CharField(max_length=500)
    full_list_report_path = models.CharField(max_length=500)
    # Automatically track the date of the analysis
    analysis_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analysis for {self.user.username} on {self.analysis_date.strftime('%Y-%m-%d')}"
    
    class Meta:
        # Order by newest first
        ordering = ['-analysis_date']
        
class ResultImage(models.Model):
    # This ForeignKey is the crucial link.
    # It connects each image to ONE AnalysisResult.
    result = models.ForeignKey(
        AnalysisResult, 
        on_delete=models.CASCADE, 
        related_name="visualizations"  # <-- This is how you get the list!
    )
    
    # This handles the file upload and stores the path
    image = models.ImageField(upload_to='analysis_images/')
    alt_text = models.CharField(max_length=250, blank=True)

    def __str__(self):
        return f"Image for {self.result.upload_filename}"
