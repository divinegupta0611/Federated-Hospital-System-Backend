from django.http import HttpResponse

def home(request):
    return HttpResponse("backend server is running")
