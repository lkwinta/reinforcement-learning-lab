# Środowisko
Zdecydowałem się na użycie środowiska LunarLander-v3, gdyż jest to klasyczne środowisko.
Ciekaweym aspektem okazał się dyskretny charakter środowiska, gdyż nanoDT nie było w pełni kompatybilne
ze środowiskiem dyskretnym, a więc musiałem dokonać kontrybucji do biblioteki :).

# Eksperci
Jako eksperta, użyłem modelu DQN który był trenowany przez 100tysięcy kroków. Wybrałem parametry z rl-zoo, które były rekomendowane dla tego środowiska.
Jako gorszego eksperta, użyłem modelu DQN który był trenowany przez 70000 kroków.

Następnie dla obu ekspertów, zebrałem `1_000_000` kroków czasowych do datasetu.

# Behavior Cloning
Następnie wytrenowałem model BC na obu datasetach. Model BC był prostym modelem MLP, który przyjmował na wejściu stan i zwracał akcję. Model był trenowany przez 100 epok, z batch size 256 i learning rate 1e-3.

```
Linear(obs_dim → 256) + ReLU
        │
        ▼
Linear(256 → 128) + ReLU
        │
        ▼
Linear(128 → act_dim)
        │
        ▼
    action scores
        │
        ▼
    argmax → action
```

# Decision Transformer
Przeprowadziłem optymalizację hiperaparametrów optuną dla DT, i finalny model trenowałem z takimi parametrami:

```
K = 60
max_ep_len = 1000
learning_rate = 1e-4
weight_decay = 0.0485
reward_scale = 13.194
```

# Wyniki:

| Model | Średnia nagroda |
| --- | --- |
| Wytrenowany ekspert (DQN) | 269.62 |
| Ekspert (DQN) - 70000 kroków | 215.85 |
| BC - pełny ekspert | 245.61 |
| BC - 70000 kroków eksperta | 165.13 |
| DT - pełny ekspert | 271.39 |
| DT - 70000 kroków eksperta | 233.95 |

Rzecz którą można łatwo zauważyć, to fakt że faktycznie DT poradził sobie lepiej niż BC, ale najciekaszą obserwacją jest 
fakt, że DT prześcignął swojego eksperta - wychwycił uchwyte struktury. 
Największą wadą DT jest duży czas treningu i predykcji, ale w zamian otrzymujemy lepsze wyniki.
