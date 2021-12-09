import io
import csv
import os.path
import datetime
today_date = datetime.date.today()
def csv_writer(data):
    if not os.path.exists('structuredInvoice/'+str(today_date)+'/'):
        os.makedirs('structuredInvoice/'+str(today_date)+'/')
    filename = 'structuredInvoice/'+str(today_date)+'/'+'InvoiceCsvData.csv'
    file_exists = os.path.isfile(filename)
    with io.open(filename,'a',encoding='utf-8') as csvfile:
        fieldnames =['ClientInfo','Amount','Items','InvoiceNumber','CompnayInfo','InvoiceDate']
       
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows([{
            'ClientInfo':data['ClientInfo'],
                           'Amount':data['Amount'],
                           'Items':data['Items'],
                           'InvoiceNumber':data['InvoiceNumber']
                           ,'CompnayInfo':data['CompanyInfo'],
                           'InvoiceDate':data['InvoiceDate']
        }])
#         print("writing complete")