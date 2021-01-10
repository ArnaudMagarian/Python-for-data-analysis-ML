from __future__ import unicode_literals
from djongo import models

# Create your models here.

class Standard(models.Model) :
	created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
	updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)
	
	name = models.CharField(max_length=20, null=True)
	pwd = models.CharField(max_length=20, null=True)
	
	def __str__(self) :
		return self.name
		
		
		
class Analyste(models.Model) :
	created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
	updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)
	
	name = models.CharField(max_length=20, null=True)
	pwd = models.CharField(max_length=20, null=True)
	
	def __str__(self) :
		return self.name
		
		
		
class Admin(models.Model) :
	created_at = models.DateTimeField(auto_now_add=True, auto_now=False)
	updated_at = models.DateTimeField(auto_now_add=False, auto_now=True)
	
	name = models.CharField(max_length=20, null=True)
	pwd = models.CharField(max_length=20, null=True)
	
	def __str__(self) :
		return self.name