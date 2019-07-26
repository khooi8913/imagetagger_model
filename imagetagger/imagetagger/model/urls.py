from django.conf.urls import url

from . import views


app_name = 'model'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^upload$', views.upload_model_image, name='upload_model_image')
]