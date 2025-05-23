# recommender_logic.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from geopy.distance import geodesic
import random
from datetime import datetime

# Variáveis globais para os modelos e dados pré-processados
df_cooperativas = pd.DataFrame()
df_produtos_features = pd.DataFrame()
df_ratings_global = pd.DataFrame() # Para armazenar todos os ratings
all_available_products = []
product_to_idx = {}
cosine_sim_matrix = None
svd_model = None

# --- DADOS DE PREFERÊNCIAS INICIAIS (SIMULADO) ---
# Em um sistema real, isso viria de um banco de dados
user_initial_preferences = {} # user_id: {'categories': ['Fruta', 'Verdura'], 'location': (lat, lon)}

def load_and_preprocess_data():
    global df_cooperativas, df_produtos_features, all_available_products, product_to_idx, cosine_sim_matrix, svd_model, df_ratings_global

    try:
        df_cooperativas_raw = pd.read_csv("produtos_cooperativas_preenchido_precos_similares.csv")
        df_produtos_features_raw = pd.read_csv("produtos.csv")
    except FileNotFoundError:
        print("ERRO: Arquivos CSV não encontrados.")
        return False

    # Preprocessing df_cooperativas_raw (simplificado)
    metadata_cols = ['Nome', 'Latitude', 'Longitude', 'Tipo_Organizacao', 'Ano_Fundacao',
                     'Certificacao_Organica_Geral', 'Selo_Agricultura_Familiar_Possui',
                     'Horario_Funcionamento_Atendimento', 'Regioes_Entrega',
                     'Formas_Pagamento_Aceitas', 'Faz_Entrega']
    product_cols_coop = [col for col in df_cooperativas_raw.columns if col not in metadata_cols]

    def clean_price(price_str):
        if isinstance(price_str, (int, float)): return float(price_str)
        if isinstance(price_str, str):
            price_str = price_str.replace(',', '.')
            try: return float(price_str)
            except ValueError: return 0.0
        return 0.0

    for col in product_cols_coop:
        df_cooperativas_raw[col] = df_cooperativas_raw[col].apply(clean_price)

    df_cooperativas_melted = df_cooperativas_raw.melt(
        id_vars=metadata_cols, value_vars=product_cols_coop,
        var_name='Nome_Produto', value_name='Preco'
    )
    df_cooperativas = df_cooperativas_melted[df_cooperativas_melted['Preco'] > 0].copy()
    df_cooperativas.rename(columns={'Nome': 'Nome_Cooperativa'}, inplace=True)
    all_available_products = sorted(df_cooperativas['Nome_Produto'].unique())

    # Preprocessing df_produtos_features
    df_produtos_features = df_produtos_features_raw.copy()
    df_produtos_features.set_index('Nome_Produto', inplace=True)
    feature_cols_for_CB = ['Categoria_Produto', 'Subcategoria_Produto', 'Perfil_Sabor_Predominante',
                           'Textura_Predominante', 'Cor_Predominante_Visual', 'Uso_Culinario_Principal']
    
    # Garantir que todas as colunas de features existam
    for col in feature_cols_for_CB:
        if col not in df_produtos_features.columns:
            df_produtos_features[col] = "N/A" # Adiciona coluna vazia se não existir

    df_produtos_features['Combined_Features'] = df_produtos_features[feature_cols_for_CB].apply(
        lambda row: ' '.join(row.astype(str).values), axis=1
    )
    
    missing_in_features = [p for p in all_available_products if p not in df_produtos_features.index]
    for p_name in missing_in_features:
        if p_name not in df_produtos_features.index:
            df_produtos_features.loc[p_name, 'Combined_Features'] = "Informacao Indisponivel"
            df_produtos_features.loc[p_name, 'Categoria_Produto'] = "Desconhecida"


    # Simular alguns ratings iniciais se df_ratings_global estiver vazio (para SVD não falhar)
    # Em um sistema real, os ratings seriam carregados de um BD
    if df_ratings_global.empty:
        print("Simulando alguns ratings iniciais para o modelo SVD...")
        num_initial_users = 10
        ratings_init_data = []
        if all_available_products:
            for uid_init in range(1, num_initial_users + 1):
                prods_to_rate_init = random.sample(all_available_products, min(5, len(all_available_products)))
                for prod_init in prods_to_rate_init:
                    ratings_init_data.append({
                        'user_id': uid_init, 
                        'item_id': prod_init, 
                        'rating': random.randint(3, 5),
                        'timestamp': datetime.now(),
                        'cooperative_name': df_cooperativas[df_cooperativas['Nome_Produto'] == prod_init]['Nome_Cooperativa'].sample(1).iloc[0] if not df_cooperativas[df_cooperativas['Nome_Produto'] == prod_init].empty else "N/A"
                    })
            df_ratings_global = pd.DataFrame(ratings_init_data)
        else:
            print("AVISO: Nenhum produto disponível, SVD não será treinado efetivamente.")


    # Treinar SVD
    if not df_ratings_global.empty:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_ratings_global[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd_model = SVD(n_factors=50, n_epochs=20, random_state=42, biased=True)
        svd_model.fit(trainset)
        print("Modelo SVD treinado.")
    else:
        svd_model = None
        print("Modelo SVD não treinado (sem ratings).")


    # Content-Based
    if all_available_products and not df_produtos_features.empty:
        # Garantir que product_features_for_tfidf use apenas produtos existentes no índice
        valid_products_for_tfidf = [p for p in all_available_products if p in df_produtos_features.index]
        if valid_products_for_tfidf:
            product_features_for_tfidf = df_produtos_features.loc[valid_products_for_tfidf, 'Combined_Features'].fillna("Informacao Indisponivel")
            if not product_features_for_tfidf.empty:
                tfidf_vectorizer = TfidfVectorizer(stop_words=None)
                tfidf_matrix = tfidf_vectorizer.fit_transform(product_features_for_tfidf)
                cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                # Atualiza product_to_idx para corresponder aos produtos em tfidf_matrix
                product_to_idx = {product_name: i for i, product_name in enumerate(valid_products_for_tfidf)}
                print("Matrizes TF-IDF e Similaridade de Cosseno computadas.")
            else:
                cosine_sim_matrix = None
                product_to_idx = {}
        else:
            cosine_sim_matrix = None
            product_to_idx = {}
    else:
        cosine_sim_matrix = None
        product_to_idx = {}
        print("Matrizes TF-IDF não computadas.")

    return True


# --- FUNÇÕES DE RECOMENDAÇÃO (ADAPTADAS DAS VERSÕES ANTERIORES) ---

def predict_cf_score(user_id, product_name, model, min_rating=1, max_rating=5):
    if model is None: return 0.5 # Retorna score neutro se modelo não existe
    # Para usuários novos não conhecidos pelo modelo SVD, a predição será a média global.
    prediction = model.predict(user_id, product_name, verbose=False)
    normalized_score = (prediction.est - min_rating) / (max_rating - min_rating)
    return normalized_score

def get_content_similarity(product1_name, product2_name):
    global cosine_sim_matrix, product_to_idx
    if cosine_sim_matrix is None or product1_name not in product_to_idx or product2_name not in product_to_idx:
        return 0.0
    idx1 = product_to_idx[product1_name]
    idx2 = product_to_idx[product2_name]
    return cosine_sim_matrix[idx1, idx2]

def predict_cb_score(user_id, target_product_name, threshold_good_rating=3.5):
    global df_ratings_global, cosine_sim_matrix, product_to_idx
    if cosine_sim_matrix is None or df_ratings_global.empty or not product_to_idx: return 0.0

    user_liked_items_df = df_ratings_global[
        (df_ratings_global['user_id'] == user_id) &
        (df_ratings_global['rating'] >= threshold_good_rating)
    ]
    if user_liked_items_df.empty: return 0.0
    user_liked_items = user_liked_items_df['item_id'].tolist()

    total_similarity = 0
    count_similar = 0
    for liked_item in user_liked_items:
        similarity = get_content_similarity(target_product_name, liked_item)
        total_similarity += similarity
        count_similar += 1
    return (total_similarity / count_similar) if count_similar > 0 else 0.0

def combine_scores(cf_score, cb_score, alpha=0.6, beta=0.4):
    return alpha * cf_score + beta * cb_score

def get_recommendation_reason(user_id, product_name, cf_score, cb_score, like_rating_threshold=4):
    # (Lógica da função get_recommendation_reason adaptada)
    # ... (Cole a lógica da sua função get_recommendation_reason aqui, adaptando as fontes de dados)
    reasons = []
    global df_ratings_global, df_produtos_features, product_to_idx, cosine_sim_matrix

    if cb_score > 0.1: # Se CB contribuiu
        user_highly_rated_products_df = df_ratings_global[
            (df_ratings_global['user_id'] == user_id) & (df_ratings_global['rating'] >= like_rating_threshold)
        ]
        if not user_highly_rated_products_df.empty:
            user_highly_rated_products = user_highly_rated_products_df['item_id'].tolist()
            best_match_similarity = 0
            best_match_product = None

            if product_name in product_to_idx:
                for liked_prod in user_highly_rated_products:
                    if liked_prod in product_to_idx:
                        sim = get_content_similarity(product_name, liked_prod)
                        if sim > best_match_similarity:
                            best_match_similarity = sim
                            best_match_product = liked_prod
                
                if best_match_product and best_match_similarity > 0.2:
                    try:
                        # Acessar 'Perfil_Sabor_Predominante' com segurança
                        target_sabor = df_produtos_features.loc[product_name, 'Perfil_Sabor_Predominante'] if product_name in df_produtos_features.index and 'Perfil_Sabor_Predominante' in df_produtos_features.columns else "características únicas"
                        target_sabor_main = target_sabor.split(',')[0] if isinstance(target_sabor, str) else "características únicas"

                        liked_sabor = df_produtos_features.loc[best_match_product, 'Perfil_Sabor_Predominante'] if best_match_product in df_produtos_features.index and 'Perfil_Sabor_Predominante' in df_produtos_features.columns else "características únicas"
                        liked_sabor_main = liked_sabor.split(',')[0] if isinstance(liked_sabor, str) else "características únicas"
                        
                        reasons.append(f"Similar a '{best_match_product}' (que você avaliou bem, com sabor {liked_sabor_main.lower()}). '{product_name}' tem sabor {target_sabor_main.lower()}.")
                    except Exception as e: # Mais genérico para pegar qualquer erro de acesso
                        # print(f"Erro ao buscar sabor para CB reason: {e}")
                        reasons.append(f"Similar a '{best_match_product}' (que você avaliou bem).")
                elif not best_match_product and user_highly_rated_products:
                     reasons.append("Baseado nas características de produtos que você gostou.")


    if cf_score > 0.7 and cb_score < 0.5 : # Se CF é forte e CB não é o principal
        reasons.append("Outros usuários com gostos parecidos também apreciaram este produto.")
    elif cf_score > 0.5 and not reasons: # Fallback se nenhuma outra razão
         reasons.append("Pode ser do seu interesse com base nas avaliações de outros.")

    if not reasons:
        # Tenta uma razão baseada na categoria se o usuário tiver preferências iniciais
        if user_id in user_initial_preferences and 'categories' in user_initial_preferences[user_id]:
            prod_cat = df_produtos_features.loc[product_name, 'Categoria_Produto'] if product_name in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else None
            if prod_cat and prod_cat in user_initial_preferences[user_id]['categories']:
                return f"Recomendado porque você demonstrou interesse na categoria '{prod_cat}'."
        return "Recomendado para você explorar novos sabores!"
    return " ".join(reasons)


def generate_personalized_recommendations(user_id, alpha=0.6, beta=0.4, top_n=10):
    global df_ratings_global, all_available_products, svd_model
    
    user_rated_items = set()
    if not df_ratings_global.empty and user_id in df_ratings_global['user_id'].unique():
        user_rated_items = set(df_ratings_global[df_ratings_global['user_id'] == user_id]['item_id'])
    
    products_to_score = [p for p in all_available_products if p not in user_rated_items]
    if not products_to_score: return []

    recommendations = []
    for product_name in products_to_score:
        cf_score = predict_cf_score(user_id, product_name, svd_model)
        cb_score = predict_cb_score(user_id, product_name) # Passa df_ratings_global implicitamente
        
        # Boost para categorias preferidas (exemplo simples)
        category_boost = 0.0
        if user_id in user_initial_preferences and 'categories' in user_initial_preferences[user_id]:
            prod_cat = df_produtos_features.loc[product_name, 'Categoria_Produto'] if product_name in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else None
            if prod_cat and prod_cat in user_initial_preferences[user_id]['categories']:
                category_boost = 0.1 # Pequeno boost

        hybrid_score = combine_scores(cf_score, cb_score, alpha, beta) + category_boost
        recommendations.append({
            'ProductName': product_name, # Nome da coluna esperado pelo HTML
            'RelevanceScore': hybrid_score, # Nome da coluna esperado pelo HTML
            'cf_score': cf_score, 
            'cb_score': cb_score
        })
    
    if not recommendations: return []
    recs_df = pd.DataFrame(recommendations)
    return recs_df.sort_values(by='RelevanceScore', ascending=False).head(top_n)


def get_popular_products_df(top_n=10):
    global df_ratings_global, all_available_products
    if df_ratings_global.empty:
        if not all_available_products: return pd.DataFrame()
        # Se não há ratings, retorna uma amostra aleatória
        sample_prods = random.sample(all_available_products, min(top_n, len(all_available_products)))
        return pd.DataFrame([{
            'ProductName': p, 
            'RelevanceScore': 0.5, # Score neutro
            'cf_score':0, 'cb_score':0} for p in sample_prods])

    product_stats = df_ratings_global.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
    min_ratings = 2 # Pelo menos 2 ratings para ser "popular"
    popular = product_stats[product_stats['count'] >= min_ratings]
    popular = popular.sort_values(by=['mean', 'count'], ascending=[False, False]).head(top_n)
    
    return pd.DataFrame([{
        'ProductName': row['item_id'],
        'RelevanceScore': row['mean'] / 5.0, # Normaliza
        'cf_score':0, 'cb_score':0
    } for _, row in popular.iterrows()])


def get_final_recommendations_with_coops(user_id, user_lat, user_lon, max_dist_km, recommendations_df, recommendation_type="personalized"):
    """Junta recomendações de produtos com informações das cooperativas e aplica filtros."""
    global df_cooperativas
    if recommendations_df.empty:
        return pd.DataFrame()

    output_recs = []
    # Usar um conjunto para garantir que cada combinação produto-cooperativa seja única na saída final,
    # mas a recomendação é por PRODUTO. Então, para cada produto recomendado, listamos as cooperativas.
    
    for _, rec_product_row in recommendations_df.iterrows():
        product_name = rec_product_row['ProductName']
        
        coops_selling_this_product = df_cooperativas[df_cooperativas['Nome_Produto'] == product_name].copy()
        if coops_selling_this_product.empty:
            continue

        coops_selling_this_product['Distance_km'] = coops_selling_this_product.apply(
            lambda r: calculate_distance_km(user_lat, user_lon, r['Latitude'], r['Longitude']), axis=1
        )
        
        coops_within_dist = coops_selling_this_product[coops_selling_this_product['Distance_km'] <= max_dist_km]
        if coops_within_dist.empty:
            continue

        # Para cada cooperativa que vende o produto dentro da distância
        for _, coop_row in coops_within_dist.sort_values(by='Distance_km').iterrows():
            reason = "Produto popular!" if recommendation_type == "popular" else \
                     get_recommendation_reason(user_id, product_name, rec_product_row['cf_score'], rec_product_row['cb_score'])
            
            output_recs.append({
                'ProductName': product_name,
                'CooperativeName': coop_row['Nome_Cooperativa'],
                'Region': coop_row['Regioes_Entrega'], # Ou outra coluna de região se tiver
                'Latitude': coop_row['Latitude'],
                'Longitude': coop_row['Longitude'],
                'Distance_km': coop_row['Distance_km'],
                'Price': coop_row['Preco'],
                'Organic': coop_row['Certificacao_Organica_Geral'] >= 1,
                'FamilyFarm': coop_row['Selo_Agricultura_Familiar_Possui'] == 1,
                'RelevanceScore': rec_product_row['RelevanceScore'], # Score do produto
                'BaseScore_SVD': rec_product_row['cf_score'], # Exemplo, pode ser o CF
                'Reason': reason
            })

    if not output_recs: return pd.DataFrame()
    
    final_df = pd.DataFrame(output_recs)
    # Queremos N produtos únicos, mostrando a melhor cooperativa (mais próxima) para cada
    # Ou podemos listar todas as cooperativas para os N melhores produtos
    # Este exemplo agrupa por produto e pega a melhor cooperativa, depois desempata pelo score do produto.
    # Se você quer N entradas (produto-cooperativa), a lógica de corte precisa ser diferente.
    
    # Ordena para que, ao remover duplicatas de ProductName, fiquemos com a melhor cooperativa (menor distância)
    final_df.sort_values(by=['ProductName', 'RelevanceScore', 'Distance_km'], ascending=[True, False, True], inplace=True) 
    # Mantém apenas a melhor oferta (menor distância) por produto recomendado, limitado por top_n_final de produtos únicos.
    # No entanto, o HTML itera sobre o que é passado. Se queremos N itens (produto-coop), o corte deve ser no final.
    
    return final_df # O HTML irá iterar sobre isso. O corte de N recomendações acontecerá no Flask.


def search_products_in_cooperatives(user_lat, user_lon, max_dist_km, product_intent=None, preferred_categories=None):
    """Busca cooperativas com base em intenção de produto e categorias, filtradas por distância."""
    global df_cooperativas, df_produtos_features
    
    relevant_coops = df_cooperativas.copy()

    # Calcular distância para todas
    relevant_coops['Distance_km'] = relevant_coops.apply(
        lambda r: calculate_distance_km(user_lat, user_lon, r['Latitude'], r['Longitude']), axis=1
    )
    relevant_coops = relevant_coops[relevant_coops['Distance_km'] <= max_dist_km]

    if not relevant_coops.empty and product_intent:
        relevant_coops = relevant_coops[relevant_coops['Nome_Produto'] == product_intent]

    if not relevant_coops.empty and preferred_categories:
        # Merge com df_produtos_features para obter a categoria do produto
        relevant_coops = pd.merge(relevant_coops, df_produtos_features[['Categoria_Produto']], 
                                  left_on='Nome_Produto', right_index=True, how='left')
        relevant_coops = relevant_coops[relevant_coops['Categoria_Produto'].isin(preferred_categories)]

    if relevant_coops.empty:
        return pd.DataFrame()

    # Adicionar colunas esperadas pelo template recommendations.html
    # BaseScore e RelevanceScore podem ser placeholder ou calculados de forma simples
    # (ex: popularidade do produto na cooperativa, ou apenas um score fixo para itens de busca direta)
    results = []
    for _, row in relevant_coops.sort_values(by=['Nome_Produto', 'Distance_km']).iterrows():
        results.append({
            'ProductName': row['Nome_Produto'],
            'CooperativeName': row['Nome_Cooperativa'],
            'Region': row['Regioes_Entrega'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Distance_km': row['Distance_km'],
            'Price': row['Preco'],
            'Organic': row['Certificacao_Organica_Geral'] >= 1,
            'FamilyFarm': row['Selo_Agricultura_Familiar_Possui'] == 1,
            'BaseScore': 0.7, # Placeholder
            'RelevanceScore': 0.8 # Placeholder
        })
    return pd.DataFrame(results)


def add_rating_logic(user_id, product_name, cooperative_name, rating_value):
    global df_ratings_global
    # Remove avaliação anterior do mesmo usuário para o mesmo produto/cooperativa, se houver
    df_ratings_global = df_ratings_global[~((df_ratings_global['user_id'] == user_id) & 
                                            (df_ratings_global['item_id'] == product_name) &
                                            (df_ratings_global['cooperative_name'] == cooperative_name))]
    
    new_rating = pd.DataFrame([{
        'user_id': user_id,
        'item_id': product_name,
        'cooperative_name': cooperative_name,
        'rating': int(rating_value),
        'timestamp': datetime.now()
    }])
    df_ratings_global = pd.concat([df_ratings_global, new_rating], ignore_index=True)
    # Idealmente, aqui você salvaria no BD e talvez dispararia um retreinamento do SVD.
    print(f"Rating adicionado/atualizado: User {user_id}, Produto {product_name}, Coop {cooperative_name}, Nota {rating_value}")
    print(f"Total ratings agora: {len(df_ratings_global)}")
    # Para fins de exemplo, vamos retreinar o SVD aqui. Em produção, isso seria mais controlado.
    if not df_ratings_global.empty:
        global svd_model
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_ratings_global[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd_model = SVD(n_factors=50, n_epochs=20, random_state=42, biased=True, verbose=False)
        svd_model.fit(trainset)
        print("Modelo SVD retreinado após novo rating.")


def get_user_ratings_df(user_id):
    global df_ratings_global
    if df_ratings_global.empty:
        return pd.DataFrame()
    user_r = df_ratings_global[df_ratings_global['user_id'] == user_id].copy()
    if not user_r.empty:
        user_r['formatted_timestamp'] = user_r['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
        # Renomear colunas para corresponder ao template my_ratings.html
        user_r.rename(columns={'item_id': 'product_name'}, inplace=True)
    return user_r.sort_values(by='timestamp', ascending=False)


def calculate_distance_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    return geodesic((lat1, lon1), (lat2, lon2)).km

load_and_preprocess_data()