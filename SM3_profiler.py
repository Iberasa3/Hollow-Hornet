import ee

#De momento esta es una versión pre-alfa, no vale para nada y solo sirve como estructura básica para que pueda trabajarlo mañana

class SM3Profiler:
    """
    Clase para implementar el Paso 2 del SM4 y el SM3 en su completitud
    Utiliza One-Class SVM para identificar zonas de 'similitud cero' respecto a las presencias
    """

    def __init__(self, kernel_type='RBF', nu=0.1, gamma=0.5):
        self.kernel_type = kernel_type
        self.nu = nu
        self.gamma = gamma
        self.model = None

    def train_ocsvm(self, presence_points, environmental_stack):
        """
        Entrena el modelo OCSVM utilizando únicamente datos de presencia.
        """
        training_features = environmental_stack.sampleRegions(
            collection=presence_points,
            scale=1000
        )
        # Basado en la literatura de Senay et al. (2013)
        self.model = ee.Classifier.libsvm(
            svmType='ONE_CLASS',
            kernelType=self.kernel_type,
            nu=self.nu,
            gamma=self.gamma
        ).train(training_features, 'class', environmental_stack.bandNames())

        return self.model

    def get_zero_similarity_mask(self, environmental_stack, aoi):
        """
        Aplica el modelo y devuelve una máscara de las zonas hostiles (valor 0).
        """
        if not self.model:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")

        # Clasificar el espacio ambiental
        similarity_map = environmental_stack.clip(aoi).classify(self.model)

        # El paper SM4 sugiere que los puntos con probabilidad 0 son las ausencias ideales
        # Invertimos: donde el mapa es 0, la máscara es 1 (permitido para pseudo-ausencias)
        return similarity_map.eq(0).selfMask()


def generate_environmental_absences(presences, predictors, aoi, num_points, seed=67):
    """
    Función de utilidad para llamar desde el Notebook.
    """
    profiler = SM3Profiler()
    profiler.train_ocsvm(presences, predictors)
    mask = profiler.get_zero_similarity_mask(predictors, aoi)

    # Muestreo final de los 'ceros' inteligentes
    absences = mask.sample(
        region=aoi,
        scale=1000,
        numPixels=num_points * 2,  # Sobremuestreo para el truncamiento
        seed=seed,
        geometries=True
    ).limit(num_points)

    return absences.map(lambda f: f.set('class', 0))