# forms.py
from django import forms
from .models import Scan

class ScanForm(forms.ModelForm):
    url = forms.URLField(
        label='SEC Filing URL',
        widget=forms.TextInput(attrs={
            'class': 'form-control form-control-lg',
            'placeholder': 'https://www.sec.gov/Archives/edgar/data/...',
            'autofocus': True,
            'style': 'height: 58px; font-size: 1.1rem;',
        })
    )

    class Meta:
        model = Scan
        fields = ['url']