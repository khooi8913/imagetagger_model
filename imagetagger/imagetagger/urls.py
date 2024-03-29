"""imagetagger URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.shortcuts import render
from registration.backends.hmac.views import RegistrationView

from .users.forms import UserRegistrationForm

urlpatterns = [
    url(r'^user/', include('django.contrib.auth.urls')),
    url(r'^accounts/register/$', RegistrationView.as_view(form_class=UserRegistrationForm)),
    url(r'^accounts/', include('registration.backends.hmac.urls')),
    url(r'^', include('imagetagger.base.urls')),
    url(r'^admin/', admin.site.urls),
    url(r'^administration/', include('imagetagger.administration.urls')),
    url(r'^annotations/', include('imagetagger.annotations.urls')),
    url(r'^images/', include('imagetagger.images.urls')),
    url(r'^users/', include('imagetagger.users.urls')),
    url(r'^tagger_messages/', include('imagetagger.tagger_messages.urls')),
    url(r'^tools/', include('imagetagger.tools.urls')),
    url(r'^model/', include('imagetagger.model.urls')),
]


def handler500(request):
    """500 error handler which includes ``request`` in the context.

    Templates: `500.html`
    Context: None
    """
    return render(request, '500.html', status=500)
