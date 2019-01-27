import numpy as np
from scipy.special import gamma
# from scipy.stats import describe
from scipy.stats import mode
import matplotlib.pyplot as plt


def extract_Sjj(S, clust, partition):
    i_clust = np.array(partition == clust, dtype=int)
    return np.dot(i_clust, np.dot(S, i_clust))


def post_sampling(S, k, d, r, s, xi, theta, nr, burn):
    # init
    epsilon = np.finfo(np.float32).eps
    K = np.array([])
    n = S.shape[0]
    B = np.eye(n)
    sum_B = np.zeros((n, n))
    nr_B = 0
    partition = np.array([i % k for i in range(n)])
#    partition = np.array([i+1 for i in range(n)])
    n_theta = len(theta)
    prior_theta = np.array([1/n_theta] * n_theta)

    # also init \Prob(B|xi)
    log_prob_B_cond_xi = len(np.unique(partition)) * np.log(xi) \
                         + np.log(gamma(xi)) - np.log(gamma(xi + n))
    for clust in np.unique(partition):
        log_prob_B_cond_xi += np.log(gamma(sum(partition == clust)))

    it = 0
    while it < nr:
        u_clust = np.unique(partition)

        # (a) posteriors on theta and sampling from it
        log_post_theta = np.array([])
        for i_theta in range(n_theta):
            theta_i = theta[i_theta]
            log_prod_term = 0.
            sum_term = 0.
            for clust in u_clust:
                n_c = sum(partition == clust)
                log_prod_term += np.log(1 + theta_i*n_c)
                sum_term += (theta_i/(1 + n_c*theta_i)*extract_Sjj(S, clust, partition))
            log_post_S = -d/2.*log_prod_term - (n+r)*d/2.*np.log(d/2.*(S.trace() - sum_term + s))
            log_post_theta = np.append(log_post_theta, log_post_S + np.log(prior_theta[i_theta] + epsilon))
        log_post_theta -= np.max(log_post_theta)
        post_theta = np.exp(log_post_theta)
        post_theta /= np.sum(post_theta)
        theta_choice = np.random.choice(theta, p=post_theta)
        prob_theta_choice = post_theta[theta == theta_choice]
#        print(prob_theta_choice)
#        print(np.log(prob_theta_choice))
        prior_theta = post_theta

        # (b) sampling of B
        # introduce possibility of a new cluster if appropriate
        it_clust = np.array([])
#        new_clust = None
        if len(u_clust) < n:
            it_clust = np.append(u_clust, np.max(u_clust)+1)
#            new_clust = it_clust[-1]
        else:
            it_clust = u_clust

        curr_log_prod_term = 0.
        curr_sum_term = 0.
        for clust in u_clust:
            curr_n_c = sum(partition == clust)
            curr_log_prod_term += np.log(1 + theta_choice*curr_n_c)
            curr_sum_term += (theta_choice/(1 + curr_n_c*theta_choice)*extract_Sjj(S, clust, partition))
            
        for i_item in range(n):      
            if it == burn:
                for i in range(n):
                    for j in range(i+1, n-1):
                        if partition[i] == partition[j]:
                            B[i, j] = 1
                            B[j, i] = 1
                
            log_post_B = np.array([])

            curr_clust = partition[i_item]
            curr_n_c = sum(partition == curr_clust)

            for clust in it_clust:
                it_partition = np.array(partition)
                it_partition[i_item] = clust
                it_n_c = sum(it_partition == clust)
                it_log_prod_term = curr_log_prod_term \
                                   - np.log(1 + theta_choice*curr_n_c) \
                                   + np.log(1 + theta_choice*(curr_n_c - 1)) \
                                   - np.log(1 + theta_choice*(it_n_c - 1)) \
                                   + np.log(1 + theta_choice*it_n_c)
                it_sum_term = curr_sum_term \
                              - (theta_choice/(1 + curr_n_c*theta_choice)*extract_Sjj(S, curr_clust, partition)) \
                              + (theta_choice/(1 + (curr_n_c-1)*theta_choice)*extract_Sjj(S, curr_clust, it_partition)) \
                              - (theta_choice/(1 + (it_n_c-1)*theta_choice)*extract_Sjj(S, clust, partition)) \
                              + (theta_choice/(1 + it_n_c*theta_choice)*extract_Sjj(S, clust, it_partition))
                it_log_prob_B_cond_xi = log_prob_B_cond_xi \
                                        - np.log(max(curr_n_c - 1, 1)) \
                                        + np.log(it_n_c) \
                                        + (len(np.unique(it_partition)) - len(np.unique(partition))) * np.log(xi)

#                stats = {
#                        "it_log_prod_term": it_log_prod_term,
#                          "S.trace()": S.trace(),
#                          "it_sum_term": it_sum_term,
#                          "it_log_prob_B_cond_xi": it_log_prob_B_cond_xi,
#                          "prob_theta_choice": prob_theta_choice}

                value = -d/2. * it_log_prod_term \
                       - (n+r)*d/2. * np.log(d/2.*(S.trace()-it_sum_term + s)) \
                       + it_log_prob_B_cond_xi \
                       + np.log(prob_theta_choice)
                log_post_B = np.append(log_post_B, value)

            log_post_B -= max(log_post_B)
            post_B = np.exp(log_post_B)
            post_B /= np.sum(post_B)
            new_clust = np.random.choice(it_clust, p=post_B)
            new_partition = np.array(partition)
            new_partition[i_item] = new_clust
            new_n_c = sum(new_partition == new_clust)
            curr_log_prod_term += (- np.log(1 + theta_choice*curr_n_c)
                                   + np.log(1 + theta_choice*(curr_n_c - 1))
                                   - np.log(1 + theta_choice*(new_n_c - 1))
                                   + np.log(1 + theta_choice*new_n_c))
            curr_sum_term += (- (theta_choice/(1 + curr_n_c*theta_choice)*extract_Sjj(S, curr_clust, partition))
                              + (theta_choice/(1 + (curr_n_c-1)*theta_choice)*extract_Sjj(S, curr_clust, new_partition)) #
                              - (theta_choice/(1 + (new_n_c-1)*theta_choice)*extract_Sjj(S, new_clust, partition))       #
                              + (theta_choice/(1 + new_n_c*theta_choice)*extract_Sjj(S, new_clust, new_partition)))
            log_prob_B_cond_xi += (- np.log(max(curr_n_c - 1, 1))
                                   + np.log(new_n_c)
                                   + (len(np.unique(new_partition)) - len(np.unique(partition))) * np.log(xi))
            
            partition = new_partition
            
            if it >= burn:
                B[i_item, :] = 0
                B[:, i_item] = 0
                B[i_item, partition == new_clust] = 1
                B[partition == new_clust, i_item] = 1
                sum_B += B
                nr_B += 1
                K = np.append(K, len(np.unique(partition)))

#            print("{}/{}".format(it+1, nr), np.unique(partition))
            it += 1
            if it == nr:
                break
#            print("---------------------------------------")

    # num_bins = 10
    plt.hist(K, facecolor='blue', alpha=0.5)
    plt.show()
    mode_k, _ = mode(K)
    print("mode_k", mode_k)
    print(partition)
    np.savetxt("B_text.csv", sum_B/nr_B, delimiter=",")

    print("END")


def main():
#    S = np.loadtxt(open("./varcovar.csv", "rb"), delimiter=",")
#    post_sampling(S, k=10, d=2, r=3, s=4, xi=0.2, theta=np.array([1., 2., 3., 4., 5.])*0.1, nr=8000, burn=1000)
    
    S = np.loadtxt(open("./S_text.csv", "rb"), delimiter=",")
    post_sampling(S, k=8, d=15, r=3, s=4, xi=0.5, theta=np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])*0.01, nr=8000, burn=1000)


main()