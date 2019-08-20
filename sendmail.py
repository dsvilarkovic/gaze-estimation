import smtplib
import os
from email.mime.text import MIMEText




email = "glupisiromasni@gmail.com"
password = "Password996/"
to = "dusansvilarkovic@gmail.com"


def sendmail_content(content):
    msg = MIMEText(content)
    msg['Subject'] = 'Report for process %d' % (os.getpid())
    msg['From'] = email
    msg['To'] = to
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    #Next, log in to the server
    server.login(email, password)

     
    server.sendmail(email, to, msg.as_string())
    server.quit()


