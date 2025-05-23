from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import recommender_logic as rl # Nosso módulo de lógica
import secrets # Para session key
from datetime import datetime # Para passar 'now' para o layout

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

DEFAULT_LAT = -15.793889
DEFAULT_LON = -47.882778
DEFAULT_MAX_DIST_KM = 20
MIN_RATINGS_FOR_PERSONALIZED = 1

# Helper para obter o ID do usuário da sessão
def get_current_user_id():
    if 'user_id' not in session:
        new_user_id_val = 1
        if not rl.df_ratings_global.empty and 'user_id' in rl.df_ratings_global.columns:
            valid_user_ids = rl.df_ratings_global['user_id'].dropna()
            if not valid_user_ids.empty:
                max_id = valid_user_ids.max()
                new_user_id_val = int(max_id) + 1
        
        session['user_id'] = new_user_id_val
        if new_user_id_val not in rl.user_initial_preferences:
            rl.user_initial_preferences[new_user_id_val] = {} 
        flash(f"Bem-vindo(a)! Você é o usuário #{session['user_id']}. Por favor, defina suas preferências iniciais ou comece a buscar!", "info")
    return int(session['user_id'])

# Helper para verificar se o usuário tem ratings
def user_has_ratings_check(user_id):
    if rl.df_ratings_global.empty or user_id not in rl.df_ratings_global['user_id'].unique():
        return False
    return not rl.df_ratings_global[rl.df_ratings_global['user_id'] == user_id].empty

@app.route('/')
def index():
    user_id = get_current_user_id()
    
    # Redireciona para preferências iniciais se não foram apresentadas OU se não há localização salva
    user_prefs = rl.user_initial_preferences.get(user_id, {})
    if not session.get('initial_prefs_presented', False) or not user_prefs.get('location'):
        return redirect(url_for('initial_preferences_route'))

    current_location = user_prefs.get('location', (DEFAULT_LAT, DEFAULT_LON))

    return render_template('index.html',
                           products_list=rl.all_available_products,
                           categories_list=sorted(rl.df_produtos_features['Categoria_Produto'].unique().tolist()) if 'Categoria_Produto' in rl.df_produtos_features.columns else [],
                           default_location=current_location,
                           default_max_distance_km=DEFAULT_MAX_DIST_KM,
                           user_id=user_id,
                           now=datetime.utcnow(),
                           page_uses_map=False, # Index não usa mapa diretamente
                           hide_initial_prefs_link=True # Já passou daqui ou está aqui
                           )

@app.route('/initial-preferences', methods=['GET'])
def initial_preferences_route():
    user_id = get_current_user_id()
    # Não marcamos 'initial_prefs_presented' aqui, mas na submissão ou pulo.
    return render_template('initial_preferences.html',
                           categories=sorted(rl.df_produtos_features['Categoria_Produto'].unique().tolist()) if 'Categoria_Produto' in rl.df_produtos_features.columns else [],
                           user_id=user_id,
                           now=datetime.utcnow(),
                           page_uses_map=False,
                           hide_initial_prefs_link=True # Está na página de prefs
                           )

@app.route('/save-initial-preferences', methods=['POST'])
def save_initial_preferences():
    user_id = get_current_user_id()
    try:
        lat = float(request.form.get('latitude'))
        lon = float(request.form.get('longitude'))
    except (TypeError, ValueError): # Se lat/lon não forem fornecidos ou inválidos
        flash("Localização inválida ou não fornecida. Por favor, tente novamente.", "error")
        return redirect(url_for('initial_preferences_route'))

    rl.user_initial_preferences[user_id] = {
        'name': request.form.get('user_name', ''), # Adicionado valor padrão
        'categories': request.form.getlist('preferred_categories'),
        'location': (lat, lon)
    }
    session['initial_prefs_presented'] = True
    flash("Preferências salvas! Agora você pode buscar ou ver recomendações 'Para Você'.", "success")
    if rl.user_initial_preferences[user_id]['categories']:
         return redirect(url_for('for_you_recommendations_route'))
    return redirect(url_for('index'))

@app.route('/skip-initial-preferences')
def skip_initial_preferences():
    user_id = get_current_user_id()
    # Garante que uma localização padrão seja definida se o usuário pular
    # e nenhuma localização foi obtida/definida ainda.
    current_prefs = rl.user_initial_preferences.get(user_id, {})
    if 'location' not in current_prefs:
        current_prefs['location'] = (DEFAULT_LAT, DEFAULT_LON)
    rl.user_initial_preferences[user_id] = current_prefs # Salva de volta

    session['initial_prefs_presented'] = True
    flash("Preferências iniciais puladas. Sua localização foi definida para um valor padrão. Você pode buscar produtos ou ver recomendações populares.", "info")
    return redirect(url_for('index'))

# ROTA DE BUSCA CORRIGIDA
@app.route('/search', methods=['POST'])
def search_route(): # Nome da função é o endpoint por padrão
    user_id = get_current_user_id()
    try:
        lat = float(request.form.get('latitude', DEFAULT_LAT))
        lon = float(request.form.get('longitude', DEFAULT_LON))
        max_dist = int(request.form.get('max_distance', DEFAULT_MAX_DIST_KM))
        product_intent = request.form.get('search_intent_product')
        # No index.html, o name é 'preferred_categories_search'
        preferred_categories = request.form.getlist('preferred_categories_search') 
    except (ValueError, TypeError):
        flash("Valores inválidos para localização ou distância.", "error")
        return redirect(url_for('index'))

    recommendations_df = rl.search_products_in_cooperatives(lat, lon, max_dist, product_intent, preferred_categories)
    
    user_ratings_dict = {}
    if user_has_ratings_check(user_id):
        user_r = rl.df_ratings_global[rl.df_ratings_global['user_id'] == user_id]
        for _, row in user_r.iterrows():
            user_ratings_dict[(row['item_id'], row['cooperative_name'])] = int(row['rating'])

    return render_template('recommendations.html',
                           recommendations=recommendations_df,
                           location=(lat, lon),
                           max_distance=max_dist,
                           product_intent=product_intent,
                           preferred_categories=preferred_categories,
                           user_id=user_id,
                           user_product_ratings=user_ratings_dict,
                           page_uses_map=True,
                           now=datetime.utcnow(),
                           hide_initial_prefs_link=True
                           )

@app.route('/for-you')
def for_you_recommendations_route():
    user_id = get_current_user_id()
    user_prefs = rl.user_initial_preferences.get(user_id, {})
    
    if 'location' not in user_prefs: # Se por algum motivo não tem localização, redireciona
        flash("Por favor, defina sua localização nas preferências iniciais ou permita o uso da localização atual.", "warning")
        return redirect(url_for('initial_preferences_route'))
        
    user_lat, user_lon = user_prefs['location']
    max_dist = user_prefs.get('max_distance_preference', DEFAULT_MAX_DIST_KM) # Poderia ser uma preferência

    user_ratings_count = rl.df_ratings_global[rl.df_ratings_global['user_id'] == user_id].shape[0] if user_has_ratings_check(user_id) else 0
    
    recommendation_type = "personalized"
    # Modifica a condição para usar categorias se os ratings forem poucos
    if user_ratings_count < MIN_RATINGS_FOR_PERSONALIZED and not user_prefs.get('categories'):
        product_recs_df = rl.get_popular_products_df(top_n=10)
        recommendation_type = "popular"
        if product_recs_df.empty:
             flash("Nenhum produto popular encontrado no momento.", "info")
    else: # Usuário tem ratings suficientes OU tem categorias preferidas
        product_recs_df = rl.generate_personalized_recommendations(user_id, top_n=10)
        if product_recs_df.empty:
            product_recs_df = rl.get_popular_products_df(top_n=10)
            recommendation_type = "popular_fallback"
            if not product_recs_df.empty:
                flash("Não foi possível gerar recomendações personalizadas. Mostrando populares.", "info")
            else:
                flash("Não encontramos recomendações para você no momento. Tente avaliar mais produtos ou buscar diretamente.", "warning")

    final_recs_df = pd.DataFrame() # Inicializa
    if not product_recs_df.empty:
        final_recs_df = rl.get_final_recommendations_with_coops(
            user_id, user_lat, user_lon, max_dist, product_recs_df, recommendation_type
        ).head(10)

    user_ratings_dict = {}
    if user_has_ratings_check(user_id):
        user_r = rl.df_ratings_global[rl.df_ratings_global['user_id'] == user_id]
        for _, row in user_r.iterrows():
            user_ratings_dict[(row['item_id'], row['cooperative_name'])] = int(row['rating'])
            
    return render_template('for_you_recommendations.html',
                           recommendations=final_recs_df,
                           location=(user_lat, user_lon),
                           max_distance=max_dist,
                           user_has_ratings=user_ratings_count > 0, # Para o template controlar o que mostrar
                           user_id=user_id,
                           user_product_ratings=user_ratings_dict,
                           page_uses_map=True,
                           now=datetime.utcnow(),
                           hide_initial_prefs_link=True
                           )

@app.route('/rate', methods=['POST'])
def rate_product():
    user_id = get_current_user_id()
    cooperative_name = request.form.get('cooperative_name')
    product_name = request.form.get('product_name')
    rating_str = request.form.get('rating')

    if not all([cooperative_name, product_name, rating_str]):
        flash("Erro ao processar avaliação. Dados incompletos.", "error")
    else:
        try:
            rating = int(rating_str)
            if not 1 <= rating <= 5:
                raise ValueError("Rating fora do intervalo válido.")
            rl.add_rating_logic(user_id, product_name, cooperative_name, rating)
            flash(f"Avaliação para '{product_name}' na '{cooperative_name}' salva!", "success")
        except ValueError:
            flash("Valor de avaliação inválido.", "error")
        except Exception as e:
            flash(f"Erro ao salvar avaliação: {e}", "error")
            app.logger.error(f"Erro ao salvar avaliação: {e}") # Log do erro no servidor

    return redirect(url_for('for_you_recommendations_route'))


@app.route('/my-ratings')
def my_ratings():
    user_id = get_current_user_id()
    ratings_df = rl.get_user_ratings_df(user_id)
    return render_template('my_ratings.html',
                           ratings=ratings_df,
                           user_id=user_id, # Para o layout
                           now=datetime.utcnow(),
                           page_uses_map=False,
                           hide_initial_prefs_link=True
                           )

@app.route('/logout')
def logout():
    user_id_logged_out = session.pop('user_id', None)
    session.pop('initial_prefs_presented', None) # Limpa o estado de apresentação
    if user_id_logged_out:
        flash(f"Você (Usuário #{user_id_logged_out}) foi desconectado.", "info")
    else:
        flash("Você foi desconectado.", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Verifica se os dados foram carregados no módulo recommender_logic
    if not rl.df_cooperativas.empty and rl.all_available_products:
        app.run(debug=True)
    else:
        print("Aplicação não iniciada. Falha ao carregar dados das cooperativas/produtos ou nenhum produto disponível.")
        if rl.df_cooperativas.empty:
            print("- df_cooperativas está vazio.")
        if not rl.all_available_products:
            print("- all_available_products está vazio.")