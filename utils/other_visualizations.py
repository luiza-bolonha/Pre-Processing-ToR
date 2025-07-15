from arulespy.arules import Transactions, apriori, parameters
from arulespy.arulesViz import plot
from rpy2.ipython.ggplot import image_png
import matplotlib.pyplot as plt


def get_rules_to_arulespy(transacoes, support):
    transacoes_sanitizadas = [
        [str(item).strip() for item in trans if item and str(item).strip() != '']
        for trans in transacoes
    ]
    item_labels = sorted({item for trans in transacoes_sanitizadas for item in trans})
    item_labels_set = set(item_labels)
    
    for trans in transacoes_sanitizadas:
        for item in trans:
            if item not in item_labels_set:
                print(f"Item não encontrado em item_labels: {item}")

    trans = Transactions.from_list(transacoes_sanitizadas, item_labels)
    rules = apriori(
        trans,
        parameter=parameters({"supp": support}),
        control=parameters({"verbose": False})
    )
    return rules

def scartter_plot_from_ranking(df):
    df_top = df.sort_values(by='quantidade', ascending=False).head(50)

    plt.figure(figsize=(12, 6))
    plt.scatter(df_top['palavra'], df_top['quantidade'], color='blue')

    plt.xticks(rotation=90)
    plt.title('Dispersão das Top 50 Palavras')
    plt.xlabel('Palavras')
    plt.ylabel('Quantidade')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
