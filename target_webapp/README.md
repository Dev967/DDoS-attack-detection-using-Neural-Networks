# Target Web Application

This application was used as target during the experiments.  
it is a flask application with logging function. The logger logs response time, Route, status and incoming IP
addresses  
for every address.  
For simulating real world applications we tried to capture as much variance as possible when deciding the routes,
their  
methods and type of resources they serve.  
The web application allows following routes:

| Route         | Method   | Description                                                                                  | Status Codes  | 
|---------------|----------|----------------------------------------------------------------------------------------------|---------------|
| /index.html   | GET      | Home page heavily loaded with CSS and Javascript                                             | 304           |
| /about.html   | GET      | Page contains only HTML and CSS, relatively faster to load                                   | 304           |
| /gallery.html | GET      | Page loaded with lots of Images, aimed to simulate high load web pages during attack         | 304           |
| /login.html   | GET/POST | Protected page, redirects once logged in                                                     | 200, 304, 401 | 
| /admin        | GET      | Protected page accessible only after succesfully logging in and when session has not expired | 200, 403      |  

