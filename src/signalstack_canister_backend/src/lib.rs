use candid::{CandidType, Deserialize, Nat};
use ic_cdk::api::management_canister::http_request::{
    CanisterHttpRequestArgument, HttpHeader, HttpResponse, TransformContext,
};
use ic_cdk::api::time;
use ic_cdk_macros::{init, post_upgrade, pre_upgrade, query, update};
use serde::Serialize;
use std::cell::RefCell;
use std::collections::HashMap;

// Constants
const TOTAL_WEIGHT: u32 = 10_000; // 100% in basis points
const REFRESH_INTERVAL: u64 = 300_000_000_000; // 5 minutes in nanoseconds
const MAX_TOKENS: usize = 20;

// Core data structures (unchanged)
#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct TokenSignal {
    pub total_score: f64,
    pub mean_reversion: f64,
    pub momentum: f64,
    pub volatility: f64,
    pub breakout: f64,
    pub ml_confidence: f64,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct TokenPrice {
    pub symbol: String,
    pub price: f64,
    pub last_updated: u64,
    pub change_24h: f64,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct RiskAssessment {
    pub risk_score: f64,
    pub volatility: f64,
    pub portfolio_volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub concentration_risk: f64,
    pub market_risk: String,
    pub recommendations: Vec<String>,
    pub anomalies_detected: bool,
    pub risk_level: String,
    pub volatility_spike: bool,
    pub market_stress: f64,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct MarketData {
    pub fear_greed_index: u32,
    pub total_market_cap: f64,
    pub btc_dominance: f64,
    pub active_cryptos: u32,
    pub market_change_24h: f64,
    pub volume_24h: f64,
    pub trending: Vec<String>,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct TokenInfo {
    pub symbol: String,
    pub balance: u64,
    pub weight: u32,
    pub current_price: f64,
    pub target_weight: f64,
    pub signal: Option<TokenSignal>,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub timestamp: u64,
    pub tx_type: String,
    pub from_token: String,
    pub to_token: String,
    pub amount: f64,
    pub gas_fee: f64,
    pub status: String,
    pub tx_hash: String,
}

#[derive(Clone, Debug, CandidType, Deserialize)]
pub struct PortfolioHistory {
    pub timestamp: u64,
    pub portfolio_value: f64,
    pub daily_return: f64,
    pub cumulative_return: f64,
}

#[derive(Default, CandidType, Deserialize)]
pub struct Portfolio {
    pub supported_tokens: Vec<String>,
    pub token_weights: HashMap<String, u32>,
    pub token_balances: HashMap<String, u64>,
    pub token_signals: HashMap<String, TokenSignal>,
    pub token_prices: HashMap<String, TokenPrice>,
    pub target_weights: HashMap<String, f64>,
    pub risk_assessment: Option<RiskAssessment>,
    pub market_data: Option<MarketData>,
    pub transactions: Vec<Transaction>,
    pub portfolio_history: Vec<PortfolioHistory>,
    pub last_update: u64,
    pub total_weight: u32,
    pub total_portfolio_value: f64,
}

// Thread-local storage
thread_local! {
    static PORTFOLIO: RefCell<Portfolio> = RefCell::new(Portfolio::default());
}

// HTTP Request handling
#[query]
async fn http_request(request: CanisterHttpRequestArgument) -> HttpResponse {
    let path = request.url.split('?').next().unwrap_or("/");

    match path {
        "/" => serve_frontend(),
        "/api/portfolio" => serve_portfolio_api(),
        "/api/health" => serve_health_api(),
        _ => HttpResponse {
            status: 404u16.into(),
            headers: vec![],
            body: b"Not Found".to_vec(),
        },
    }
}

// Helper function to create HTTP responses
fn create_response(
    status: u16,
    content_type: &str,
    body: Vec<u8>,
    cors: bool,
) -> HttpResponse {
    let mut headers = vec![HttpHeader {
        name: "Content-Type".to_string(),
        value: content_type.to_string(),
    }];

    if cors {
        headers.push(HttpHeader {
            name: "Access-Control-Allow-Origin".to_string(),
            value: "*".to_string(),
        });
    }

    HttpResponse {
        status: status.into(),
        headers,
        body,
    }
}

fn serve_frontend() -> HttpResponse {
    let html_content = include_str!("../../../frontend/index.html");
    create_response(200, "text/html", html_content.as_bytes().to_vec(), false)
}


fn serve_portfolio_api() -> HttpResponse {
    let portfolio_data = PORTFOLIO.with(|p| {
        serde_json::to_string(&p.borrow()).unwrap_or_else(|_| "{}".to_string())
    });
    create_response(200, "application/json", portfolio_data.into_bytes(), true)
}

fn serve_health_api() -> HttpResponse {
    let health_data = get_health_status();
    create_response(
        200,
        "application/json",
        serde_json::to_string(&health_data)
            .unwrap_or_else(|_| "{}".to_string())
            .into_bytes(),
        true,
    )
}

// Helper functions for JSON serialization
fn serialize_tokens(portfolio: &Portfolio) -> String {
    let mut tokens = Vec::new();

    for symbol in &portfolio.supported_tokens {
        let balance = *portfolio.token_balances.get(symbol).unwrap_or(&0);
        let price = portfolio.token_prices.get(symbol).map(|p| p.price).unwrap_or(0.0);
        let weight = *portfolio.token_weights.get(symbol).unwrap_or(&0);
        let target_weight = *portfolio.target_weights.get(symbol).unwrap_or(&0.0);
        let signal = portfolio.token_signals.get(symbol);

        let token_json = format!(
            r#"{{
                "symbol": "{}",
                "price": {},
                "balance": {},
                "weight": {},
                "targetWeight": {},
                "signal": {{
                    "total_score": {},
                    "momentum": {},
                    "volatility": {}
                }}
            }}"#,
            symbol,
            price,
            balance as f64 / 1_000_000.0,
            weight as f64 / 100.0,
            target_weight,
            signal.map(|s| s.total_score).unwrap_or(0.0),
            signal.map(|s| s.momentum).unwrap_or(0.0),
            signal.map(|s| s.volatility).unwrap_or(0.0)
        );

        tokens.push(token_json);
    }

    tokens.join(",")
}

fn serialize_risk_assessment(risk: &Option<RiskAssessment>) -> String {
    match risk {
        Some(r) => format!(
            r#"{{
                "risk_score": {},
                "risk_level": "{}",
                "volatility": {}
            }}"#,
            r.risk_score,
            r.risk_level,
            r.volatility
        ),
        None => "null".to_string(),
    }
}

fn serialize_market_data(market: &Option<MarketData>) -> String {
    match market {
        Some(m) => format!(
            r#"{{
                "fear_greed_index": {},
                "btc_dominance": {}
            }}"#,
            m.fear_greed_index,
            m.btc_dominance
        ),
        None => "null".to_string(),
    }
}

fn serialize_transactions(transactions: &[Transaction]) -> String {
    let mut tx_strings = Vec::new();

    for tx in transactions.iter().take(5) {
        let tx_json = format!(
            r#"{{
                "id": "{}",
                "timestamp": {},
                "tx_type": "{}",
                "amount": {},
                "status": "{}"
            }}"#,
            tx.id,
            tx.timestamp,
            tx.tx_type,
            tx.amount,
            tx.status
        );
        tx_strings.push(tx_json);
    }

    tx_strings.join(",")
}

#[init]
fn init() {
    ic_cdk::println!("🚀 SignalStack DeFi Portfolio Canister with Frontend initialized");

    let default_tokens = vec![
        ("BTC".to_string(), 3500),
        ("ETH".to_string(), 3000),
        ("ADA".to_string(), 1500),
        ("DOT".to_string(), 1000),
        ("USDC".to_string(), 1000),
    ];

    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();
        for (symbol, weight) in default_tokens {
            portfolio.supported_tokens.push(symbol.clone());
            portfolio.token_weights.insert(symbol.clone(), weight);
            portfolio.token_balances.insert(symbol.clone(), 0);
            portfolio.total_weight += weight;
        }
        portfolio.last_update = time();
    });

    let _ = refresh_market_data();
}

// Signal Generation Functions
fn generate_mock_signal(symbol: &str, base_time: u64) -> TokenSignal {
    let seed = symbol.len() as f64 + (base_time as f64 / 1_000_000_000.0);
    let hash = ((seed * 9999.0) % 1000.0) / 1000.0;

    TokenSignal {
        total_score: (hash - 0.5) * 4.0,
        mean_reversion: (hash * 1.3 % 1.0 - 0.5) * 2.0,
        momentum: (hash * 1.7 % 1.0 - 0.5) * 2.0,
        volatility: (hash * 2.1 % 1.0) * 2.0,
        breakout: (hash * 0.9 % 1.0 - 0.5) * 2.0,
        ml_confidence: 0.5 + (hash * 0.4),
    }
}

fn generate_mock_price(symbol: &str, base_time: u64) -> TokenPrice {
    let base_prices = match symbol {
        "BTC" => 45000.0,
        "ETH" => 3000.0,
        "ADA" => 1.20,
        "DOT" => 25.0,
        "USDC" => 1.00,
        _ => 100.0,
    };

    let seed = (base_time as f64 / 1_000_000_000.0) % 100.0;
    let variation = (seed - 50.0) * 0.002;
    let price = base_prices * (1.0 + variation);

    TokenPrice {
        symbol: symbol.to_string(),
        price,
        last_updated: base_time,
        change_24h: variation * 100.0,
    }
}

fn calculate_target_weights(signals: &HashMap<String, TokenSignal>) -> HashMap<String, f64> {
    let mut weights = HashMap::new();

    let base_weights = vec![
        ("BTC", 35.0),
        ("ETH", 30.0),
        ("ADA", 15.0),
        ("DOT", 10.0),
        ("USDC", 10.0),
    ];

    for (symbol, base_weight) in base_weights {
        let signal_adjustment = signals.get(symbol)
            .map(|s| s.total_score * 2.0)
            .unwrap_or(0.0);

        let adjusted_weight = (base_weight + signal_adjustment).max(5.0).min(50.0);
        weights.insert(symbol.to_string(), adjusted_weight);
    }

    let total: f64 = weights.values().sum();
    if total > 0.0 {
        for weight in weights.values_mut() {
            *weight = (*weight / total) * 100.0;
        }
    }

    weights
}

fn calculate_risk_assessment(
    weights: &HashMap<String, f64>,
    _prices: &HashMap<String, TokenPrice>,
    signals: &HashMap<String, TokenSignal>
) -> RiskAssessment {
    let base_volatility = 25.0;

    let max_weight: f64 = weights.values().fold(0.0, |a, &b| a.max(b));
    let concentration_risk = if max_weight > 40.0 {
        (max_weight - 40.0) * 2.0 + 50.0
    } else {
        50.0 - (40.0 - max_weight)
    };

    let avg_volatility: f64 = signals.values().map(|s| s.volatility.abs()).sum::<f64>() / signals.len() as f64;
    let portfolio_volatility = base_volatility + avg_volatility * 5.0;

    let risk_score = (concentration_risk * 0.3 + portfolio_volatility * 0.4 + avg_volatility * 10.0) / 10.0;

    let market_risk = if risk_score > 7.0 { "High" }
                     else if risk_score > 4.0 { "Medium" }
                     else { "Low" };

    let recommendations = vec![
        "📊 Monitor concentration risk in top holdings".to_string(),
        "⚖️ Consider rebalancing if volatility increases".to_string(),
        "🔍 Watch for market anomalies and correlation changes".to_string(),
    ];

    RiskAssessment {
        risk_score,
        volatility: avg_volatility * 10.0,
        portfolio_volatility,
        max_drawdown: 35.0,
        sharpe_ratio: 0.8,
        concentration_risk,
        market_risk: market_risk.to_string(),
        recommendations,
        anomalies_detected: false,
        risk_level: market_risk.to_string(),
        volatility_spike: avg_volatility > 1.5,
        market_stress: (risk_score / 10.0).min(1.0),
    }
}

fn generate_market_data(base_time: u64) -> MarketData {
    let time_factor: f64 = (base_time as f64 / 1_000_000_000.0) % 100.0;

    MarketData {
        fear_greed_index: (50.0 + (time_factor - 50.0) * 0.8) as u32,
        total_market_cap: 2.1e12,
        btc_dominance: 42.5,
        active_cryptos: 23847,
        market_change_24h: (time_factor - 50.0) * 0.1,
        volume_24h: 1.2e11,
        trending: vec![
            String::from("BTC"),
            String::from("ETH"),
            String::from("ADA"),
            String::from("DOT"),
            String::from("SOL"),
        ],
    }
}

// Core canister functions
#[update]
fn refresh_market_data() -> Result<String, String> {
    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();
        let current_time = time();

        for symbol in &portfolio.supported_tokens.clone() {
            let signal = generate_mock_signal(symbol, current_time);
            portfolio.token_signals.insert(symbol.clone(), signal);

            let price = generate_mock_price(symbol, current_time);
            portfolio.token_prices.insert(symbol.clone(), price);
        }

        portfolio.target_weights = calculate_target_weights(&portfolio.token_signals);

        portfolio.risk_assessment = Some(calculate_risk_assessment(
            &portfolio.target_weights,
            &portfolio.token_prices,
            &portfolio.token_signals
        ));

        portfolio.market_data = Some(generate_market_data(current_time));

        let mut total_value = 0.0;
        for (symbol, balance) in &portfolio.token_balances {
            if let Some(price) = portfolio.token_prices.get(symbol) {
                total_value += (*balance as f64 / 1_000_000.0) * price.price;
            }
        }
        portfolio.total_portfolio_value = total_value;

        portfolio.last_update = current_time;

        Ok("Market data refreshed successfully".to_string())
    })
}

#[update]
fn add_token(symbol: String, weight: u32) -> Result<(), String> {
    if symbol.len() > 10 {
        return Err("Token symbol too long".to_string());
    }

    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();

        if portfolio.supported_tokens.len() >= MAX_TOKENS {
            return Err(format!("Maximum {} tokens supported", MAX_TOKENS));
        }

        if portfolio.token_weights.contains_key(&symbol) {
            return Err("Token already exists".to_string());
        }

        if portfolio.total_weight + weight > TOTAL_WEIGHT {
            return Err(format!(
                "Adding this token exceeds total allowed weight of {}",
                TOTAL_WEIGHT
            ));
        }

        portfolio.supported_tokens.push(symbol.clone());
        portfolio.token_weights.insert(symbol.clone(), weight);
        portfolio.token_balances.insert(symbol.clone(), 0);
        portfolio.total_weight += weight;

        let current_time = time();
        let signal = generate_mock_signal(&symbol, current_time);
        let price = generate_mock_price(&symbol, current_time);

        portfolio.token_signals.insert(symbol.clone(), signal);
        portfolio.token_prices.insert(symbol.clone(), price);

        Ok(())
    })
}

#[update]
fn remove_token(symbol: String) -> Result<(), String> {
    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();

        match portfolio.token_weights.get(&symbol) {
            Some(weight) => {
                portfolio.total_weight -= *weight;
                portfolio.token_weights.remove(&symbol);
                portfolio.token_balances.remove(&symbol);
                portfolio.token_signals.remove(&symbol);
                portfolio.token_prices.remove(&symbol);
                portfolio.target_weights.remove(&symbol);
                portfolio.supported_tokens.retain(|x| x != &symbol);
                Ok(())
            }
            None => Err("Token not found".to_string()),
        }
    })
}

#[update]
fn update_token_balance(symbol: String, balance: u64) -> Result<(), String> {
    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();

        if portfolio.token_balances.contains_key(&symbol) {
            portfolio.token_balances.insert(symbol, balance);
            Ok(())
        } else {
            Err("Token not supported".to_string())
        }
    })
}

#[update]
fn rebalance_portfolio(target_weights: HashMap<String, f64>) -> Result<String, String> {
    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();
        let current_time = time();

        let total_weight: f64 = target_weights.values().sum();
        if (total_weight - 100.0).abs() > 1.0 {
            return Err("Target weights must sum to approximately 100%".to_string());
        }

        let tx_id = format!("tx_{}", current_time);
        let transaction = Transaction {
            id: tx_id.clone(),
            timestamp: current_time,
            tx_type: "rebalance".to_string(),
            from_token: "PORTFOLIO".to_string(),
            to_token: "REBALANCED".to_string(),
            amount: portfolio.total_portfolio_value,
            gas_fee: 0.005,
            status: "completed".to_string(),
            tx_hash: format!("0x{:x}", current_time % 0xFFFFFFFF),
        };
        portfolio.transactions.insert(0, transaction);

        if portfolio.transactions.len() > 50 {
            portfolio.transactions.truncate(50);
        }

        for (symbol, weight) in target_weights {
            portfolio.target_weights.insert(symbol, weight);
        }

        Ok(format!("Portfolio rebalanced successfully. Transaction ID: {}", tx_id))
    })
}

// Query functions
#[query]
fn get_health_status() -> HashMap<String, String> {
    let mut status = HashMap::new();

    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        let current_time = time();
        let time_since_update = current_time.saturating_sub(portfolio.last_update);

        status.insert("status".to_string(), "healthy".to_string());
        status.insert("timestamp".to_string(), current_time.to_string());
        status.insert("last_update".to_string(), portfolio.last_update.to_string());
        status.insert("time_since_update_ns".to_string(), time_since_update.to_string());
        status.insert("supported_tokens".to_string(), portfolio.supported_tokens.len().to_string());
        status.insert("total_portfolio_value".to_string(), portfolio.total_portfolio_value.to_string());
        status.insert("signals".to_string(), "active".to_string());
        status.insert("prices".to_string(), "active".to_string());
        status.insert("risk_manager".to_string(), "active".to_string());
    });

    status
}

#[query]
fn get_signals_and_weights() -> (HashMap<String, TokenSignal>, HashMap<String, f64>, u64) {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        (
            portfolio.token_signals.clone(),
            portfolio.target_weights.clone(),
            portfolio.last_update
        )
    })
}

#[query]
fn get_current_prices() -> (HashMap<String, TokenPrice>, u64) {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        (portfolio.token_prices.clone(), portfolio.last_update)
    })
}

#[query]
fn get_risk_assessment() -> (Option<RiskAssessment>, u64) {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        (portfolio.risk_assessment.clone(), portfolio.last_update)
    })
}

#[query]
fn get_market_overview() -> (Option<MarketData>, u64) {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        (portfolio.market_data.clone(), portfolio.last_update)
    })
}

#[query]
fn get_portfolio_summary() -> Vec<TokenInfo> {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();

        portfolio.supported_tokens.iter().map(|symbol| {
            TokenInfo {
                symbol: symbol.clone(),
                balance: *portfolio.token_balances.get(symbol).unwrap_or(&0),
                weight: *portfolio.token_weights.get(symbol).unwrap_or(&0),
                current_price: portfolio.token_prices.get(symbol).map(|p| p.price).unwrap_or(0.0),
                target_weight: *portfolio.target_weights.get(symbol).unwrap_or(&0.0),
                signal: portfolio.token_signals.get(symbol).cloned(),
            }
        }).collect()
    })
}

#[query]
fn get_recent_transactions(limit: Option<usize>) -> Vec<Transaction> {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        let max_limit = limit.unwrap_or(10).min(50);

        portfolio.transactions.iter()
            .take(max_limit)
            .cloned()
            .collect()
    })
}

#[query]
fn find_most_underweight() -> Option<(String, f64)> {
    PORTFOLIO.with(|p| {
        let portfolio = p.borrow();
        let total_value = portfolio.total_portfolio_value;
        if total_value == 0.0 {
            return None;
        }

        let mut max_deficit = 0.0;
        let mut result: Option<String> = None;

        for symbol in &portfolio.supported_tokens {
            let current_balance = *portfolio.token_balances.get(symbol).unwrap_or(&0) as f64 / 1_000_000.0;
            let current_price = portfolio.token_prices.get(symbol).map(|p| p.price).unwrap_or(0.0);
            let current_value = current_balance * current_price;

            let target_weight = *portfolio.target_weights.get(symbol).unwrap_or(&0.0) / 100.0;
            let target_value = total_value * target_weight;

            if current_value < target_value {
                let deficit = target_value - current_value;
                if deficit > max_deficit {
                    max_deficit = deficit;
                    result = Some(symbol.clone());
                }
            }
        }

        result.map(|symbol| (symbol, max_deficit))
    })
}

#[pre_upgrade]
fn pre_upgrade() {
    ic_cdk::println!("Preparing for upgrade - saving state");
}

#[post_upgrade]
fn post_upgrade() {
    ic_cdk::println!("Post upgrade - reinitializing canister");

    let default_tokens = vec![
        ("BTC".to_string(), 3500),
        ("ETH".to_string(), 3000),
        ("ADA".to_string(), 1500),
        ("DOT".to_string(), 1000),
        ("USDC".to_string(), 1000),
    ];

    PORTFOLIO.with(|p| {
        let mut portfolio = p.borrow_mut();
        portfolio.supported_tokens.clear();
        portfolio.token_weights.clear();
        portfolio.token_balances.clear();
        portfolio.token_signals.clear();
        portfolio.token_prices.clear();
        portfolio.target_weights.clear();
        portfolio.transactions.clear();
        portfolio.portfolio_history.clear();
        portfolio.total_weight = 0;

        for (symbol, weight) in default_tokens {
            portfolio.supported_tokens.push(symbol.clone());
            portfolio.token_weights.insert(symbol.clone(), weight);
            portfolio.token_balances.insert(symbol.clone(), 0);
            portfolio.total_weight += weight;
        }
        portfolio.last_update = time();
    });

    let _ = refresh_market_data();
}