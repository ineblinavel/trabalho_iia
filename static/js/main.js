document.addEventListener('DOMContentLoaded', function () {
    const getLocationBtn = document.getElementById('getLocationBtn');
    const latitudeInput = document.getElementById('latitude');
    const longitudeInput = document.getElementById('longitude');
    const locationStatus = document.getElementById('locationStatus');

    if (getLocationBtn) {
        getLocationBtn.addEventListener('click', function () {
            if (navigator.geolocation) {
                locationStatus.textContent = 'Obtendo sua localização...';
                getLocationBtn.disabled = true; // Desabilita o botão enquanto busca

                navigator.geolocation.getCurrentPosition(
                    function (position) {
                        const lat = position.coords.latitude;
                        const lon = position.coords.longitude;

                        latitudeInput.value = lat.toFixed(6); // 6 casas decimais para precisão
                        longitudeInput.value = lon.toFixed(6);
                        locationStatus.textContent = 'Localização obtida com sucesso!';
                        locationStatus.style.color = '#4CAF50'; // Verde para sucesso
                        getLocationBtn.disabled = false; // Reabilita o botão
                    },
                    function (error) {
                        let errorMessage = 'Erro ao obter localização: ';
                        switch (error.code) {
                            case error.PERMISSION_DENIED:
                                errorMessage += "Permissão negada. Por favor, permita o acesso à sua localização.";
                                break;
                            case error.POSITION_UNAVAILABLE:
                                errorMessage += "Informação de localização indisponível.";
                                break;
                            case error.TIMEOUT:
                                errorMessage += "Tempo limite excedido ao tentar obter a localização.";
                                break;
                            case error.UNKNOWN_ERROR:
                                errorMessage += "Um erro desconhecido ocorreu.";
                                break;
                        }
                        locationStatus.textContent = errorMessage;
                        locationStatus.style.color = '#d9534f'; // Vermelho para erro
                        getLocationBtn.disabled = false; // Reabilita o botão
                    },
                    {
                        enableHighAccuracy: true, // Tenta obter a melhor precisão possível
                        timeout: 10000,          // Tempo máximo em ms para tentar obter a localização
                        maximumAge: 0            // Não usar um resultado em cache, obter um novo
                    }
                );
            } else {
                locationStatus.textContent = 'Geolocalização não é suportada por este navegador.';
                locationStatus.style.color = '#f0ad4e'; // Laranja para aviso
            }
        });
    }
});