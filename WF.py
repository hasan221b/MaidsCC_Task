from csv import writer

def write_row(file, L):
    with open(file, 'a') as f_object:

        writer_object = writer(f_object)


        writer_object.writerow(L)

        f_object.close()
