from django import forms
from .models import Plant
class PlantForm(forms.ModelForm):
    class Meta:
        model = Plant
        fields = [
            'name', 'W_max', 'k', 'n', 'b', 'max_moves', 'Yield', 'planting_cost', 'revenue'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter plant name'}),
            'W_max': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max width'}),
            'k': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Growth rate constant'}),
            'n': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Exponent'}),
            'b': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Parameter b'}),
            'max_moves': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Max moves'}),
            'Yield': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Yield'}),
            'planting_cost': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Planting cost'}),
            'revenue': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Revenue'}),
        }

    # Override default validation if needed
    def clean(self):
        cleaned_data = super().clean()
        # Add any custom validations or defaults here
        return cleaned_data


class SimulationForm(forms.Form):
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Simulation Name', 'id': 'unique_name_id',})    )
    length = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Length (cm)',"min":1,"value":100})
    )
    startDate = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date',"min": "2022-09-30", "max": "2024-03-31","value":"2022-09-30"})
    )
    stepSize = forms.IntegerField(
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Step Size', "min":1,"value":24})
    )
    harvestType = forms.ChoiceField(
        choices=[('max_yield', 'Max Yield'), ('max_quality', 'Max Quality'), ('earliest', 'Earliest')],
        widget=forms.Select(attrs={'class': 'form-select'})
    )


