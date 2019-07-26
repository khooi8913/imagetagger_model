from django import forms


class ModelUploadForm(forms.Form):
    image = forms.FileField(label='image')
    model = forms.FileField(label='model')