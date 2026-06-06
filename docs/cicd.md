# CI/CD

Le workflow GitHub Actions execute trois etapes sur `main`, `dev` et sur chaque pull request.

1. installation des dependances Python ;
2. execution des tests unitaires ;
3. build de l'image Docker de l'API.

Le deploiement n'est pas encore automatise. La prochaine etape naturelle consiste a publier l'image API sur un registre et a brancher Streamlit Community Cloud ou une cible equivalente.
