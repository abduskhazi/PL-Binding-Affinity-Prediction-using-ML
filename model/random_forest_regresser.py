from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import data_bakery as bakery
import reproducibility
import matplotlib
import matplotlib.pyplot as plt
import random
import numpy as np
import sys
sys.path.append('../')
import RotationForest.RotationForest as rf

regressor = None

def func(args):
    population_part, X, y, regressor = args
    regressor.set_params(n_jobs=1) # To avoid multiprocessing warning.
    X_backup = np.copy(X)

    initial_score = r2_score(y, regressor.predict(X))
    # Copy X_validate to prevent modification.

    score = []
    n = 0
    for feature_selection in population_part:
        shuffle_indexes = [id for id, e in enumerate(feature_selection) if e == 0]
        for i in shuffle_indexes:
            np.random.shuffle(X[:, i])
        score_after_shuffle = r2_score(y, regressor.predict(X))

        n = n + 1
        print("R2 score reference = ", r2_score(y, regressor.predict(X_backup)), "Finished - ", n, end="\r")
        X = np.copy(X_backup)
        num_features_selected = sum(feature_selection)  # The optimization tries to find the minima.
        score += [(initial_score - score_after_shuffle) * num_features_selected]

    return score

def main():
    # Firstly ...
    ExecutionID = None
    if len(sys.argv) > 1:
        ExecutionID = int(sys.argv[1])
    ExecutionID = reproducibility.reproduce(ExecutionID)

    # Obtain data for our random forest regressor.
    X, y, features, weights = bakery.bake_train_Xy()
    print("X.shape =", X.shape)
    print("y.shape =", y.shape)

    weights = np.array(weights, dtype='int64')
    X = np.concatenate((X, weights[:, np.newaxis]), axis=1)
    print("X concatenated with weights")
    print("    X.shape =", X.shape)
    print("    y.shape =", y.shape)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2)

    weights_train = np.array(X_train[:, -1], dtype='int64')
    weights_validate = np.array(X_validate[:, -1], dtype='int64')

    X = np.delete(X, -1, axis=1)
    X_train = np.delete(X_train, -1, axis=1)
    X_validate = np.delete(X_validate, -1, axis=1)
    print("Before weight duplication")
    print("    X_train.shape =", X_train.shape)
    print("    y_train.shape =", y_train.shape)

    X_train = np.repeat(X_train, weights_train, axis=0)
    y_train = np.repeat(y_train, weights_train, axis=0)
    print("After weight duplication")
    print("    X_train.shape =", X_train.shape)
    print("    y_train.shape =", y_train.shape)

    rotation = False
    if rotation:
        print("Fitting the Rotation Forest Regressor...")
        regressor = rf.RotationForest(n_trees=100, n_features=15, sample_prop=0.5, bootstrap=True) # features = partitions here.
    else:
        print("Fitting the Random Forest Regressor...")
        regressor = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1)

    regressor.fit(X_train, y_train)
    print("Fitting completed.")

    y_pred = regressor.predict(X_validate)
    if not rotation:
        print("oob score = ", regressor.oob_score_)
    print("Validation r2 score = ", r2_score(y_validate, y_pred))
    print("Training r2 score = ", r2_score(y_train, regressor.predict(X_train)))

    #Plotting to visualize the accuracy of our model.
    fig = plt.figure()
    plt.scatter(y_validate, y_pred, 1)
    plt.xlabel("y_validate")
    plt.ylabel("y_validate_pred")
    plt.title("Execution ID = " + str(ExecutionID))
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('accuracy_validate.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.scatter(y_train, regressor.predict(X_train), 1)
    plt.xlabel("y_train")
    plt.ylabel("y_train_pred")
    plt.title("Execution ID = " + str(ExecutionID))
    plt.plot(range(2,14), range(2,14), '--')
    fig.savefig('accuracy_train.png', dpi=fig.dpi)

    # TO check the importance protein features in our model -
    # We shuffle all protein feature columns one by one. Then we check the R^2 score.
    # Similar process is repeated for the ligand features.
    X_validate_backup = np.copy(X_validate)

    for i in range(55):
        np.random.shuffle(X_validate[:,i])
    print("r2 score (shuffled protein columns) =", r2_score(y_validate, regressor.predict(X_validate)))

    X_validate = np.copy(X_validate_backup)

    for i in range(55, X.shape[1]):
        np.random.shuffle(X_validate[:,i])
    print("r2 score (shuffled ligand columns) =", r2_score(y_validate, regressor.predict(X_validate)))

    # Getting the original X_validate for further checking
    X_validate = np.copy(X_validate_backup)

    # Gini Importance (Sorting in the decreasing order of importance)
    impt_indices = regressor.feature_importances_.argsort()[::-1]

    print("Gini impt ->", [regressor.feature_importances_[i] for i in impt_indices[:5]])

    fig = plt.figure()
    fig.subplots_adjust(left=0.35)
    ax = fig.add_subplot(111) #fig.add_axes([0,0,1,1])
    feature_names = [features[i] for i in impt_indices[:15]]
    importances = [regressor.feature_importances_[i] for i in impt_indices[:15]]
    y_pos = np.arange(len(importances))
    ax.barh(y_pos, importances, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Execution ID = " + str(ExecutionID))
    ax.set_title("Gini Importances of features.")
    fig.savefig('Gini_importance.png', dpi=fig.dpi)

    print("Important features based on decrease in entropy (Gini Importance).", "Check Gini_importance.png")
    for i in impt_indices[:30]:
        print(features[i])

    if True:
        print("Calculating importance of features using permuation importance...")
        # Permutation Importance
        from sklearn.inspection import permutation_importance
        impt = permutation_importance(regressor, X_validate, y_validate, n_repeats=30, n_jobs=-1)
        impt_indices = impt.importances_mean.argsort()[::-1]

        print("Permuation impt ->", [impt.importances_mean[i] for i in impt_indices[:5]])

        fig = plt.figure()
        fig.subplots_adjust(left=0.35)
        ax = fig.add_subplot(111) #fig.add_axes([0,0,1,1])
        feature_names = [features[i] for i in impt_indices[:15]]
        importances = [impt.importances_mean[i] for i in impt_indices[:15]]
        y_pos = np.arange(len(importances))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Execution ID = " + str(ExecutionID))
        ax.set_title("Permutation Importances of features.")
        fig.savefig('Permutation_importance.png', dpi=fig.dpi)

        print("Important features based on permutation.", "Check Permutation_importance.png")
        for i in impt_indices[:30]:
            print(features[i])

    def random_forest_score(population , X, y):
        print()

        import multiprocessing as mp
        num_cores = mp.cpu_count()
        if len(population) >= num_cores:
            pool = mp.Pool(num_cores)
            chunks = np.array_split(population, num_cores)
            work_chunks = []
            for c in chunks:
                work_chunks += [(c, np.copy(X), np.copy(y), regressor)]
            out = pool.map(func, work_chunks)
        else:
            out = [func((population, X, y, regressor))]

        score = []
        for partial_result in out:
            score += partial_result

        return score

    if False:
        # Checking genetic algorithms with random forest regressor.
        from genetic_model import genetic_algorithm

        n_iter = 100
        n_bits = X.shape[1]
        n_pop = n_bits * 6  # 100
        r_cross = 0.9
        r_mut = 1.0 / float(n_bits)

        X_validate = np.copy(X_validate_backup)
        # perform the genetic algorithm search
        print("Random forest regressor - Starting genetic algorithm")
        best, score = genetic_algorithm(random_forest_score, X_validate, y_validate, n_bits, n_iter, n_pop, r_cross, r_mut,
                                        name="rf_genetic_multiobjective_elitism")
        print('Done!')
        print('f(%s) = %f' % (best, score))

        X_validate = np.copy(X_validate_backup)
        shuffle_indexes = [id for id, e in enumerate(best) if e == 0]
        for i in shuffle_indexes:
            np.random.shuffle(X_validate[:, i])
        print("R2 score = ", r2_score(y_validate, regressor.predict(X_validate)))

    print("Program finished.")


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()
