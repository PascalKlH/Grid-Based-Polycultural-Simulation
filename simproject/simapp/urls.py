''
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='accordion_view'),  # Home page
    path('api/get_simulation_data/', views.get_simulation_data, name='get_simulation_data'),
    path('manage-plants/', views.manage_plants, name='manage_plants'),
    path('accordion-view/', views.accordion_view, name='accordion_view'),
    path('list-simulations/', views.list_simulations, name='list_simulations'),
    path('run-simulation/', views.run_simulation, name='run_simulation'),
]
