
def frequency_filter(target_company, list_companys, key_words) -> list:
    name = []
    for company in list_companys:
        if company == target_company:
            continue
        n = 0
        for word in target_company.lower().split():
            if (word in key_words) and (word in company.lower().split()):
                n += 1
        if n > 0:
            name.append(company)

    return name
