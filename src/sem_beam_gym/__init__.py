"""SEM-Beam-Gym: среда Gymnasium для коррекции луча SEM при сетевой и вибрационной помехах.

Модули:
    disturbances - физические блоки помех (источники вибрации, здание, колонна, сеть, датчики)
    scenarios    - Scenario, Randomized, типичные сценарии, моделирование помех
    env          - среда BeamEnv
    baselines    - без коррекции, линейный фильтр
    train, evaluate, plots, gallery, visualize, linear_memory, main - конвейер исследования
"""
