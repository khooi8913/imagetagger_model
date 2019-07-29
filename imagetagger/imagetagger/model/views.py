from django.contrib.auth.decorators import login_required
from django.template.response import TemplateResponse
from django.views.decorators.http import require_POST

from imagetagger.model.forms import ModelUploadForm

from imagetagger.model.utils import process_prediction, makeb64image


def _create_respond(request, image=None, error=None):
    return TemplateResponse(request, 'model/index.html', {
        'model_upload_form': ModelUploadForm(),
        'preview_image': image,
        'error': error
    })


@login_required
def index(request):
    return _create_respond(request)


@login_required
@require_POST
def upload_model_image(request):
    form = ModelUploadForm(data=request.POST, files=request.FILES)
    image = None
    error = None
    if form.is_valid():
        # return the output image after the prediction
        savedimgfile = process_prediction(request.FILES['image'],
                                   request.FILES['model'])
        # instad of saving the image at the moment,
        # it will be returned as base64 image src
        image = makeb64image(savedimgfile)
    else:
        error = form.errors

    return _create_respond(request, image=image, error=error)
