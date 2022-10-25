import models.siamese_network as sn
from configs.model_conf import model_config


def get_predict(target_company, filtered_companys) -> list[tuple[str, float]]:
    model = sn.get_siamese_model(8, 100, 128)
    model.load_weights(model_config['weights'])
    similarity = [model.predict([target_company, company]) for company in filtered_companys]

    return sorted(list(zip(filtered_companys, similarity)), key=lambda tup: tup[1], reverse=True)
