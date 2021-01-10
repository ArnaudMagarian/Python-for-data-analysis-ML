from django.contrib import admin
from health_data import models 
from .models import Standard
from .models import Analyste
from .models import Admin
# Register your models here.

admin.site.register(Standard)
admin.site.register(Analyste)
admin.site.register(Admin)