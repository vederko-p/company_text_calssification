from predict_company import get_predict
import sys
import time

def main(name):
    list_company = get_predict(name)
    recomend_company = [company for company, similarity in list_company if (similarity > 0.7)]
    print()
    print()
    if not recomend_company:
        print('Нет похожих компаний')
    else:
        print("Список похожих компаний: ", recomend_company)

    return recomend_company


if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv[1:][0])
    # main('Pirelli Tire Llc')
    print("--- Время работы программы:  %s секунд ---" % round((time.time() - start_time), 1))



