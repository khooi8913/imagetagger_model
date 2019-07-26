from django.contrib.auth.decorators import login_required
from django.template.response import TemplateResponse
from django.views.decorators.http import require_POST

from imagetagger.model.forms import ModelUploadForm


def respond(request, image=None, error=None):
    return TemplateResponse(request, 'model/index.html', {
        'model_upload_form': ModelUploadForm(),
        'preview_image': image,
        'error': error
    })


@login_required
def index(request):
    return respond(request)


@login_required
@require_POST
def upload_model_image(request):
    form = ModelUploadForm(data=request.POST, files=request.FILES)
    image = None
    error = None
    if form.is_valid():
        # make prediction and output the image
        image = "/static/symbols/logo.png"
    else:
        error = form.errors

    return respond(request, image=image, error=error)
