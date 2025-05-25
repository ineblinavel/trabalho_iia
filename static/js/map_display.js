document.addEventListener('DOMContentLoaded', function () {
    const mapElement = document.getElementById('map');
    if (!mapElement) {
        console.error("Map element with ID 'map' not found!");
        return;
    }

    const userLat = parseFloat(mapElement.dataset.userLat);
    const userLon = parseFloat(mapElement.dataset.userLon);

    let recommendationsData = [];
    try {
        if (mapElement.dataset.recommendations) {
            recommendationsData = JSON.parse(mapElement.dataset.recommendations);
        }
    } catch (e) {
        console.error("Error parsing recommendations data:", e);
    }

    if (isNaN(userLat) || isNaN(userLon)) {
        console.error("User location data is invalid or missing.");
        return;
    }

    const map = L.map('map').setView([userLat, userLon], 11);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '© <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
    }).addTo(map);

    // --- Cria um ícone vermelho customizado para a localização do usuário ---
    const redIcon = new L.Icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png', // URL para um ícone de marcador vermelho
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png', // Sombra padrão
        iconSize: [25, 41],    // Tamanho do ícone
        iconAnchor: [12, 41],  // Ponto do ícone que corresponderá à localização do marcador
        popupAnchor: [1, -34], // Ponto a partir do qual o popup deve abrir em relação ao iconAnchor
        shadowSize: [41, 41]   // Tamanho da sombra
    });

    L.marker([userLat, userLon], { icon: redIcon }).addTo(map)
        .bindPopup('<b>Sua Localização</b>');

    if (recommendationsData && recommendationsData.length > 0) {
        const markers = [];
        recommendationsData.forEach(function (rec) {
            const lat = parseFloat(rec.Latitude);
            const lon = parseFloat(rec.Longitude);
            const coopName = rec.CooperativeName;
            const productName = rec.ProductName;

            if (typeof lat === 'number' && !isNaN(lat) &&
                typeof lon === 'number' && !isNaN(lon)) {

                const marker = L.marker([lat, lon]).addTo(map)
                    .bindPopup(`<b>${coopName}</b><br>Produto: ${productName}`);
                markers.push(marker);
            } else {
                console.warn("Pulando recomendação devido a coordenadas inválidas:", rec);
            }
        });
        
    }
});