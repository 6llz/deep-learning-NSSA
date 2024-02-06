import csv

out = open('KDDTest-21.csv', 'w', newline='')
csv_writer = csv.writer(out, dialect='excel')

f = open("KDDTest-21.txt", "r")
for line in f.readlines():
    line = line.replace(',', '\t')  # 将每行的逗号替换成空格
    list = line.split()  # 将字符串转为列表，从而可以按单元格写入csv
    csv_writer.writerow(list)