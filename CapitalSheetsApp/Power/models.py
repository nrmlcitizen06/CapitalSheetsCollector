from django.db import models

# Create your models here.

class Scan(models.Model):
    url = models.URLField()
    result = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Scan {self.id} - {self.created_at}"