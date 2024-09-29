/*

For much of this project, we used: 

https://www.amazon.co.uk/dp/0521721628?ref=cm_sw_r_apin_dp_CVE4EQZDXZZNC7XPADK6

and video tutorials like: 
- https://www.youtube.com/watch?v=5e2yBSefpgo&pp=ygUebW9udGUgY2Fyb2xvIG9wdGlvbiBwcmljZXIgY3Bw
- https://www.youtube.com/watch?v=I0ogWOuf9Tc&pp=ygUebW9udGUgY2Fyb2xvIG9wdGlvbiBwcmljZXIgY3Bw

for mathematics, we consulted Dr. Cherednik and Dr. Rozansky from UNC Chapel Hill. 

This project was made on Sept 29, 2024. Led by TA Ajay Misra.

*/

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <numeric>
#include <functional>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using json = nlohmann::json;
using namespace boost::gregorian;
using namespace boost::accumulators;

// Callback function for cURL
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append((char*)contents, total_size);
    return total_size;
}

enum class OptionType { Call, Put, DigitalCall, DigitalPut };
enum class BarrierType { None, UpAndOut, DownAndOut, UpAndIn, DownAndIn };

struct OptionParams {
    double S0;    // Initial stock price
    double K;     // Strike price
    double r;     // Risk-free rate
    double q;     // Dividend yield
    double sigma; // Volatility
    double T;     // Time to maturity
    OptionType type;
    BarrierType barrierType;
    double barrierLevel;
};

class MonteCarloOptionPricer {
private:
    OptionParams params;
    int numPaths;
    int numSteps;
    std::vector<double> historicalPrices;
    std::vector<std::thread> threads;
    std::vector<double> resultsPerThread;

    // Thread-local random number generators
    std::vector<std::mt19937> generators;
    std::vector<std::normal_distribution<>> distributions;

public:
    MonteCarloOptionPricer(const OptionParams& params, int numPaths, int numSteps, int numThreads)
        : params(params), numPaths(numPaths), numSteps(numSteps) {
        generators.resize(numThreads);
        distributions.resize(numThreads);
        resultsPerThread.resize(numThreads, 0.0);

        for (int i = 0; i < numThreads; ++i) {
            generators[i] = std::mt19937(std::random_device{}());
            distributions[i] = std::normal_distribution<>(0.0, 1.0);
        }
    }

    bool fetchHistoricalData(const std::string& symbol, const date& start_date, const date& end_date) {
        // ... (implementation remains the same)
    }

    void estimateParameters() {
        accumulator_set<double, stats<tag::mean, tag::variance>> acc;

        for (size_t i = 1; i < historicalPrices.size(); ++i) {
            double return_ = std::log(historicalPrices[i] / historicalPrices[i-1]);
            acc(return_);
        }

        double mean_return = mean(acc);
        double var_return = variance(acc);

        // Annualize parameters
        params.sigma = std::sqrt(var_return * 252);
        params.r = mean_return * 252 + 0.5 * var_return * 252;  // Risk-neutral drift
    }

    std::vector<double> generatePath(std::mt19937& gen, std::normal_distribution<>& dist) {
        /*
        Stochastic Path Generation using Geometric Brownian Motion (GBM)
        
        The GBM is described by the stochastic differential equation (SDE):
        dS = μS dt + σS dW
        
        Where:
        μ = r - q (risk-neutral drift)
        σ = volatility
        dW = Wiener process increment (standard normal random variable * sqrt(dt))
        
        The solution to this SDE (applying Itô's lemma) gives the stock price at time t:
        S(t) = S(0) * exp((μ - 0.5σ^2)t + σW(t))
        
        For discrete time steps, we use:
        S(t+Δt) = S(t) * exp((μ - 0.5σ^2)Δt + σ√(Δt)Z)
        
        Where Z ~ N(0,1) is a standard normal random variable.
        */
        
        std::vector<double> path(numSteps + 1, params.S0);
        double dt = params.T / numSteps;
        double drift = params.r - params.q - 0.5 * std::pow(params.sigma, 2);
        double vol = params.sigma * std::sqrt(dt);

        for (int i = 1; i <= numSteps; ++i) {
            double Z = dist(gen);
            path[i] = path[i-1] * std::exp(drift * dt + vol * Z);
        }

        return path;
    }

    double evaluatePayoff(const std::vector<double>& path) {
        /*
        Option Payoff Evaluation
        
        1. For standard options:
           Call payoff: max(S(T) - K, 0)
           Put payoff: max(K - S(T), 0)
        
        2. For digital options:
           Digital Call payoff: 1 if S(T) > K, 0 otherwise
           Digital Put payoff: 1 if S(T) < K, 0 otherwise
        
        3. For barrier options:
           Payoff is contingent on whether the barrier condition is met
           "In" options pay off only if the barrier is touched
           "Out" options pay off only if the barrier is not touched
        */
        
        double finalPrice = path.back();
        bool barrierHit = false;

        if (params.barrierType != BarrierType::None) {
            for (const auto& price : path) {
                if ((params.barrierType == BarrierType::UpAndOut || params.barrierType == BarrierType::UpAndIn) 
                    && price >= params.barrierLevel) {
                    barrierHit = true;
                    break;
                } else if ((params.barrierType == BarrierType::DownAndOut || params.barrierType == BarrierType::DownAndIn) 
                           && price <= params.barrierLevel) {
                    barrierHit = true;
                    break;
                }
            }
        }

        auto standardPayoff = [&](bool isCall) {
            return isCall ? std::max(finalPrice - params.K, 0.0) : std::max(params.K - finalPrice, 0.0);
        };

        auto digitalPayoff = [&](bool isCall) {
            return (isCall && finalPrice > params.K) || (!isCall && finalPrice < params.K) ? 1.0 : 0.0;
        };

        auto barrierCondition = [&]() {
            return params.barrierType == BarrierType::None ||
                   (params.barrierType == BarrierType::UpAndIn && barrierHit) ||
                   (params.barrierType == BarrierType::DownAndIn && barrierHit) ||
                   (params.barrierType == BarrierType::UpAndOut && !barrierHit) ||
                   (params.barrierType == BarrierType::DownAndOut && !barrierHit);
        };

        if (barrierCondition()) {
            switch (params.type) {
                case OptionType::Call:
                case OptionType::Put:
                    return standardPayoff(params.type == OptionType::Call);
                case OptionType::DigitalCall:
                case OptionType::DigitalPut:
                    return digitalPayoff(params.type == OptionType::DigitalCall);
            }
        }
        return 0.0;
    }

    void simulatePaths(int threadId, int pathsPerThread) {
        /*
        Monte Carlo Simulation
        
        1. Generate multiple price paths
        2. Calculate the payoff for each path
        3. Take the average of these payoffs
        4. Discount the average payoff back to present value
        
        The option price is given by:
        V = e^(-rT) * E[Payoff]
        
        Where E[Payoff] is the expected payoff under the risk-neutral measure,
        estimated by the average of simulated payoffs.
        */
        
        double sumPayoffs = 0.0;
        double sumPayoffsSquared = 0.0;

        for (int i = 0; i < pathsPerThread; ++i) {
            std::vector<double> path = generatePath(generators[threadId], distributions[threadId]);
            double payoff = evaluatePayoff(path);
            sumPayoffs += payoff;
            sumPayoffsSquared += payoff * payoff;
        }

        resultsPerThread[threadId] = sumPayoffs;
    }

    std::pair<double, double> priceOption() {
        int numThreads = std::thread::hardware_concurrency();
        int pathsPerThread = numPaths / numThreads;

        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(&MonteCarloOptionPricer::simulatePaths, this, i, pathsPerThread);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        double totalSum = std::accumulate(resultsPerThread.begin(), resultsPerThread.end(), 0.0);
        double price = std::exp(-params.r * params.T) * totalSum / numPaths;
        
        // Calculate standard error
        double sumSquaredPayoffs = std::inner_product(resultsPerThread.begin(), resultsPerThread.end(),
                                                      resultsPerThread.begin(), 0.0,
                                                      std::plus<>(),
                                                      [pathsPerThread](double a, double b) { return a * b / pathsPerThread; });
        double variance = (sumSquaredPayoffs / numThreads - std::pow(price, 2)) / (numPaths - 1);
        double standardError = std::sqrt(variance / numPaths);

        return {price, standardError};
    }

    // Greek calculations using finite difference methods
    std::map<std::string, double> calculateGreeks() {
        std::map<std::string, double> greeks;
        double h = 0.01 * params.S0;  // Small change in stock price
        double basePrice = priceOption().first;

        // Delta: ∂V/∂S ≈ (V(S+h) - V(S-h)) / (2h)
        params.S0 += h;
        double priceUp = priceOption().first;
        params.S0 -= 2*h;
        double priceDown = priceOption().first;
        params.S0 += h;  // Reset to original value
        greeks["Delta"] = (priceUp - priceDown) / (2*h);

        // Gamma: ∂²V/∂S² ≈ (V(S+h) - 2V(S) + V(S-h)) / h²
        greeks["Gamma"] = (priceUp - 2*basePrice + priceDown) / (h*h);

        // Theta: -∂V/∂t ≈ -(V(t+Δt) - V(t)) / Δt
        double originalT = params.T;
        params.T -= 1.0/365.0;  // One day change
        double priceYesterday = priceOption().first;
        params.T = originalT;
        greeks["Theta"] = -(priceYesterday - basePrice) * 365;

        // Vega: ∂V/∂σ ≈ (V(σ+Δσ) - V(σ)) / Δσ
        double originalSigma = params.sigma;
        params.sigma += 0.01;
        priceUp = priceOption().first;
        params.sigma = originalSigma;
        greeks["Vega"] = (priceUp - basePrice) / 0.01;

        // Rho: ∂V/∂r ≈ (V(r+Δr) - V(r)) / Δr
        double originalR = params.r;
        params.r += 0.0001;
        priceUp = priceOption().first;
        params.r = originalR;
        greeks["Rho"] = (priceUp - basePrice) / 0.0001;

        return greeks;
    }
};

int main() {
    OptionParams params = {
        100.0,  // S0
        100.0,  // K
        0.05,   // r
        0.02,   // q
        0.2,    // sigma
        1.0,    // T
        OptionType::Call,
        BarrierType::None,
        0.0     // barrierLevel (unused for standard options)
    };

    int numPaths = 1'000'000;
    int numSteps = 252;

    MonteCarloOptionPricer pricer(params, numPaths, numSteps, std::thread::hardware_concurrency());

    // Fetch historical data (replace with actual dates)
    if (!pricer.fetchHistoricalData("AAPL", date(2022,1,1), date(2023,1,1))) {
        std::cerr << "Failed to fetch historical data" << std::endl;
        return 1;
    }

    pricer.estimateParameters();

    auto start = std::chrono::high_resolution_clock::now();
    auto [price, standardError] = pricer.priceOption();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Option Price: " << price << " ± " << standardError << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    auto greeks = pricer.calculateGreeks();
    std::cout << "\nGreeks:" << std::endl;
    for (const auto& [greek, value] : greeks) {
        std::cout << greek << ": " << value << std::endl;
    }

    return 0;
}
