# recommender_logic.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from geopy.distance import geodesic
import random
from datetime import datetime
import os
import json # Para salvar e carregar user_initial_preferences

# Variáveis globais para os modelos e dados pré-processados
df_cooperativas = pd.DataFrame()
df_produtos_features = pd.DataFrame()
df_ratings_global = pd.DataFrame() # Para armazenar todos os ratings
all_available_products = []
product_to_idx = {}
cosine_sim_matrix = None
svd_model = None

# --- DADOS DE PREFERÊNCIAS INICIAIS (SIMULADO E AGORA PERSISTENTE) ---
user_initial_preferences = {} # user_id: {'categories': ['Fruta', 'Verdura'], 'location': (lat, lon)}

# Nomes dos arquivos de persistência
RATINGS_FILE = 'data/user_ratings.csv'
PREFERENCES_FILE = 'data/user_preferences.json' # Usamos JSON para a estrutura mais complexa

def load_and_preprocess_data():
    global df_cooperativas, df_produtos_features, all_available_products, product_to_idx, cosine_sim_matrix, svd_model, df_ratings_global, user_initial_preferences

    # Garante que o diretório 'data' exista
    os.makedirs('data', exist_ok=True)

    try:
        df_cooperativas_raw = pd.read_csv("produtos_cooperativas_preenchido_precos_similares.csv")
        df_produtos_features_raw = pd.read_csv("produtos.csv")
    except FileNotFoundError:
        print("ERRO: Arquivos CSV de base não encontrados. Certifique-se de que 'produtos_cooperativas_preenchido_precos_similares.csv' e 'produtos.csv' estão no local correto.")
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
    
    for col in feature_cols_for_CB:
        if col not in df_produtos_features.columns:
            df_produtos_features[col] = "N/A"

    df_produtos_features['Combined_Features'] = df_produtos_features[feature_cols_for_CB].apply(
        lambda row: ' '.join(row.astype(str).values), axis=1
    )
    
    missing_in_features = [p for p in all_available_products if p not in df_produtos_features.index]
    for p_name in missing_in_features:
        if p_name not in df_produtos_features.index:
            df_produtos_features.loc[p_name, 'Combined_Features'] = "Informacao Indisponivel"
            df_produtos_features.loc[p_name, 'Categoria_Produto'] = "Desconhecida"

    # --- CARREGAR RATINGS E PREFERÊNCIAS PERSISTIDOS ---
    if os.path.exists(RATINGS_FILE):
        try:
            # Use parse_dates para carregar a coluna 'timestamp' como datetime
            df_ratings_global = pd.read_csv(RATINGS_FILE, parse_dates=['timestamp'])
            print(f"Ratings carregados de {RATINGS_FILE}. Total: {len(df_ratings_global)}")
        except Exception as e:
            print(f"Erro ao carregar ratings de {RATINGS_FILE}: {e}. Iniciando com DataFrame vazio.")
            df_ratings_global = pd.DataFrame()
    else:
        print(f"Arquivo de ratings '{RATINGS_FILE}' não encontrado. Iniciando com DataFrame vazio.")
        df_ratings_global = pd.DataFrame()

    if os.path.exists(PREFERENCES_FILE):
        try:
            with open(PREFERENCES_FILE, 'r') as f:
                loaded_prefs = json.load(f)
                # Converte chaves de str para int, pois IDs de usuário são int
                # E converte a lista de localização de volta para tupla
                user_initial_preferences = {int(k): {'name': v.get('name', ''), 
                                                     'categories': v.get('categories', []), 
                                                     'location': tuple(v['location']) if isinstance(v.get('location'), list) else None} 
                                            for k, v in loaded_prefs.items()}
            print(f"Preferências de usuário carregadas de {PREFERENCES_FILE}.")
        except Exception as e:
            print(f"Erro ao carregar preferências de usuário de {PREFERENCES_FILE}: {e}. Iniciando com preferências vazias.")
            user_initial_preferences = {}
    else:
        print(f"Arquivo de preferências '{PREFERENCES_FILE}' não encontrado. Iniciando com preferências vazias.")
        user_initial_preferences = {}
            
    # Simular alguns ratings iniciais se df_ratings_global ainda estiver vazio (após tentar carregar)
    if df_ratings_global.empty and all_available_products:
        print("Gerando e simulando alguns ratings iniciais para o modelo SVD para atender ao requisito de 5000 linhas...")
        
        # Aumentar o número de usuários simulados e/ou ratings por usuário para atingir 5000+ linhas
        num_initial_users_simulated = 500 
        min_ratings_per_user = 10 
        
        ratings_init_data = []
        for uid_init in range(1, num_initial_users_simulated + 1):
            num_prods_to_rate = min(min_ratings_per_user, len(all_available_products))
            
            if num_prods_to_rate == 0:
                continue

            prods_to_rate_init = random.sample(all_available_products, num_prods_to_rate)
            
            for prod_init in prods_to_rate_init:
                coop_name = "N/A"
                possible_coops = df_cooperativas[df_cooperativas['Nome_Produto'] == prod_init]['Nome_Cooperativa']
                if not possible_coops.empty:
                    coop_name = possible_coops.sample(1).iloc[0]
                
                ratings_init_data.append({
                    'user_id': uid_init, 
                    'item_id': prod_init, 
                    'rating': random.randint(1, 5),
                    'timestamp': datetime.now(),
                    'cooperative_name': coop_name
                })
        df_ratings_global = pd.DataFrame(ratings_init_data)
        save_ratings_to_csv() # Salva os ratings simulados
        print(f"Gerados {len(df_ratings_global)} ratings simulados e salvos em {RATINGS_FILE}.")
    elif df_ratings_global.empty:
        print("AVISO: Nenhum produto disponível, ratings simulados não gerados.")


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
        valid_products_for_tfidf = [p for p in all_available_products if p in df_produtos_features.index]
        if valid_products_for_tfidf:
            product_features_for_tfidf = df_produtos_features.loc[valid_products_for_tfidf, 'Combined_Features'].fillna("Informacao Indisponivel")
            if not product_features_for_tfidf.empty:
                tfidf_vectorizer = TfidfVectorizer(stop_words=None)
                tfidf_matrix = tfidf_vectorizer.fit_transform(product_features_for_tfidf)
                cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
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

# --- FUNÇÕES DE PERSISTÊNCIA ---
def save_ratings_to_csv():
    global df_ratings_global
    df_to_save = df_ratings_global.copy()
    df_to_save['timestamp'] = df_to_save['timestamp'].apply(lambda x: x.isoformat()) # Correção para isoformat
    df_to_save.to_csv(RATINGS_FILE, index=False)
    print(f"Ratings salvos em {RATINGS_FILE}.")


def save_preferences_to_json():
    global user_initial_preferences
    serializable_prefs = {str(k): {'name': v.get('name', ''), 
                                   'categories': v.get('categories', []), 
                                   'location': list(v['location']) if 'location' in v and v['location'] is not None else None} 
                          for k, v in user_initial_preferences.items()}
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(serializable_prefs, f, indent=4)
    print(f"Preferências de usuário salvas em {PREFERENCES_FILE}.")

# --- FUNÇÕES DE RECOMENDAÇÃO (com Opção 1 para razão) ---

def predict_cf_score(user_id, product_name, model, min_rating=1, max_rating=5):
    if model is None: return 0.5
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
                        target_categoria = df_produtos_features.loc[product_name, 'Categoria_Produto'] if product_name in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else ""
                        liked_categoria = df_produtos_features.loc[best_match_product, 'Categoria_Produto'] if best_match_product in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else ""

                        if target_categoria and target_categoria == liked_categoria:
                             reasons.append(f"Similar ao '{best_match_product}' (que você avaliou bem), ambos da categoria '{target_categoria}'.")
                        else:
                             reasons.append(f"Similar ao '{best_match_product}' (que você avaliou bem), com base em suas características gerais.")
                    except Exception as e:
                        reasons.append(f"Similar ao '{best_match_product}' (que você avaliou bem).")
                elif not best_match_product and user_highly_rated_products:
                     reasons.append("Baseado nas características de produtos que você gostou.")


    if cf_score > 0.7 and cb_score < 0.5 :
        reasons.append("Outros usuários com gostos parecidos com os seus também apreciaram este produto.")
    elif cf_score > 0.5 and not reasons:
         reasons.append("Pode ser do seu interesse com base nas avaliações de outros usuários.")

    if user_id in user_initial_preferences and 'categories' in user_initial_preferences[user_id]:
        prod_cat = df_produtos_features.loc[product_name, 'Categoria_Produto'] if product_name in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else None
        if prod_cat and prod_cat in user_initial_preferences[user_id]['categories']:
            if not reasons or "você demonstrou interesse na categoria" not in reasons[0]:
                reasons.append(f"Recomendado porque você demonstrou interesse na categoria '{prod_cat}'.")


    if not reasons:
        return "Recomendado para você explorar novos sabores e produtores locais!"
    return " ".join(reasons)


def generate_personalized_recommendations(user_id, alpha=0.6, beta=0.4, top_n=10):
    global df_ratings_global, all_available_products, svd_model
    
    user_rated_items = set()
    if not df_ratings_global.empty and user_id in df_ratings_global['user_id'].unique():
        user_rated_items = set(df_ratings_global[df_ratings_global['user_id'] == user_id]['item_id'])
    
    products_to_score = [p for p in all_available_products if p not in user_rated_items]
    if not products_to_score: return pd.DataFrame()

    recommendations = []
    for product_name in products_to_score:
        cf_score = predict_cf_score(user_id, product_name, svd_model)
        cb_score = predict_cb_score(user_id, product_name)
        
        category_boost = 0.0
        if user_id in user_initial_preferences and 'categories' in user_initial_preferences[user_id]:
            prod_cat = df_produtos_features.loc[product_name, 'Categoria_Produto'] if product_name in df_produtos_features.index and 'Categoria_Produto' in df_produtos_features.columns else None
            if prod_cat and prod_cat in user_initial_preferences[user_id]['categories']:
                category_boost = 0.1

        hybrid_score = combine_scores(cf_score, cb_score, alpha, beta) + category_boost
        recommendations.append({
            'ProductName': product_name,
            'RelevanceScore': hybrid_score,
            'cf_score': cf_score,
            'cb_score': cb_score
        })
    
    if not recommendations: return pd.DataFrame()
    recs_df = pd.DataFrame(recommendations)
    return recs_df.sort_values(by='RelevanceScore', ascending=False).head(top_n)


def get_popular_products_df(top_n=10):
    global df_ratings_global, all_available_products
    if df_ratings_global.empty or df_ratings_global['item_id'].nunique() < 2:
        if not all_available_products: return pd.DataFrame()
        sample_prods = random.sample(all_available_products, min(top_n, len(all_available_products)))
        return pd.DataFrame([{
            'ProductName': p, 
            'RelevanceScore': 0.5,
            'cf_score':0, 'cb_score':0} for p in sample_prods])

    product_stats = df_ratings_global.groupby('item_id')['rating'].agg(['mean', 'count']).reset_index()
    min_ratings = 2
    popular = product_stats[product_stats['count'] >= min_ratings]
    popular = popular.sort_values(by=['mean', 'count'], ascending=[False, False]).head(top_n)
    
    return pd.DataFrame([{
        'ProductName': row['item_id'],
        'RelevanceScore': row['mean'] / 5.0,
        'cf_score':0, 'cb_score':0
    } for _, row in popular.iterrows()])


def get_final_recommendations_with_coops(user_id, user_lat, user_lon, max_dist_km, recommendations_df, recommendation_type="personalized"):
    """Junta recomendações de produtos com informações das cooperativas e aplica filtros."""
    global df_cooperativas
    if recommendations_df.empty:
        return pd.DataFrame()

    output_recs = []
    
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

        for _, coop_row in coops_within_dist.sort_values(by='Distance_km').iterrows():
            if recommendation_type == "popular":
                reason = "Um dos produtos mais populares da plataforma!"
            elif recommendation_type == "popular_fallback":
                 reason = "Este é um produto popular, pois não conseguimos gerar recomendações personalizadas ainda."
            else: # recommendation_type == "personalized"
                reason = get_recommendation_reason(user_id, product_name, rec_product_row['cf_score'], rec_product_row['cb_score'])
            
            output_recs.append({
                'ProductName': coop_row['Nome_Produto'],
                'CooperativeName': coop_row['Nome_Cooperativa'],
                'Region': coop_row['Regioes_Entrega'],
                'Latitude': coop_row['Latitude'],
                'Longitude': coop_row['Longitude'],
                'Distance_km': coop_row['Distance_km'],
                'Price': coop_row['Preco'],
                'Organic': coop_row['Certificacao_Organica_Geral'] >= 1,
                'FamilyFarm': coop_row['Selo_Agricultura_Familiar_Possui'] == 1,
                'Horario_Funcionamento_Atendimento': coop_row['Horario_Funcionamento_Atendimento'],
                'Formas_Pagamento_Aceitas': coop_row['Formas_Pagamento_Aceitas'],
                'Faz_Entrega': coop_row['Faz_Entrega'] == 1,
                'RelevanceScore': rec_product_row['RelevanceScore'],
                'Reason': reason
            })

    if not output_recs: return pd.DataFrame()
    
    final_df = pd.DataFrame(output_recs)
    final_df.sort_values(by=['ProductName', 'RelevanceScore', 'Distance_km'], ascending=[True, False, True], inplace=True) 
    
    return final_df


def search_products_in_cooperatives(user_lat, user_lon, max_dist_km, product_intent=None, preferred_categories=None, organic_filter=False, family_farm_filter=False):
    """Busca cooperativas com base em intenção de produto e categorias, filtradas por distância e novos filtros."""
    global df_cooperativas, df_produtos_features
    
    relevant_coops = df_cooperativas.copy()

    relevant_coops['Distance_km'] = relevant_coops.apply(
        lambda r: calculate_distance_km(user_lat, user_lon, r['Latitude'], r['Longitude']), axis=1
    )
    relevant_coops = relevant_coops[relevant_coops['Distance_km'] <= max_dist_km]

    if not relevant_coops.empty and product_intent:
        relevant_coops = relevant_coops[relevant_coops['Nome_Produto'] == product_intent]

    if not relevant_coops.empty and preferred_categories:
        relevant_coops = pd.merge(relevant_coops, df_produtos_features[['Categoria_Produto']], 
                                  left_on='Nome_Produto', right_index=True, how='left')
        relevant_coops = relevant_coops[relevant_coops['Categoria_Produto'].isin(preferred_categories)]

    # Aplicar novos filtros
    if not relevant_coops.empty and organic_filter:
        relevant_coops = relevant_coops[relevant_coops['Certificacao_Organica_Geral'] >= 1]
    
    if not relevant_coops.empty and family_farm_filter:
        relevant_coops = relevant_coops[relevant_coops['Selo_Agricultura_Familiar_Possui'] == 1]

    if relevant_coops.empty:
        return pd.DataFrame()

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
            'Horario_Funcionamento_Atendimento': row['Horario_Funcionamento_Atendimento'],
            'Formas_Pagamento_Aceitas': row['Formas_Pagamento_Aceitas'],
            'Faz_Entrega': row['Faz_Entrega'] == 1,
        })
    return pd.DataFrame(results)


def add_rating_logic(user_id, product_name, cooperative_name, rating_value):
    global df_ratings_global
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
    
    save_ratings_to_csv() # Salva após adicionar/atualizar
    
    if not df_ratings_global.empty:
        global svd_model
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df_ratings_global[['user_id', 'item_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        svd_model = SVD(n_factors=50, n_epochs=20, random_state=42, biased=True, verbose=False)
        svd_model.fit(trainset)
        print("Modelo SVD retreinado após novo rating.")
    else:
        svd_model = None
        print("Modelo SVD desativado (sem ratings).")


def get_user_ratings_df(user_id):
    global df_ratings_global
    if df_ratings_global.empty:
        return pd.DataFrame()
    user_r = df_ratings_global[df_ratings_global['user_id'] == user_id].copy()
    if not user_r.empty:
        user_r['formatted_timestamp'] = user_r['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
        user_r.rename(columns={'item_id': 'product_name'}, inplace=True)
    return user_r.sort_values(by='timestamp', ascending=False)


def calculate_distance_km(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float('inf')
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Chama a função de carregamento na inicialização
load_and_preprocess_data()