"""website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from cellphone import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home),
    path('operation/', views.operation),
    path('data/', views.get_data),
    path('login/', views.login),
    path('register/', views.register),
    path('rating/', views.rating),
    path('rating2/', views.rating2),
    path('cellphone_avg_rate/', views.cellphone_avg_rate),
    path('favorite_cell_phone_of_users/', views.favorite_cell_phone_of_users),
    path('amount_of_cellphone_ratings/', views.amount_of_cellphone_ratings),
    path('better_cellphone/', views.better_cellphone),
    path('market_share/', views.market_share),
    path('market_share_operating_system/', views.market_share_operating_system),
    path('avg_sex_M/', views.avg_sex_M),
    path('avg_sex_F/', views.avg_sex_F),
    path('top_elder/', views.top_elder),
]
