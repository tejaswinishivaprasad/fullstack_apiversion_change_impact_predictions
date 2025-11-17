//package org.impactframework.api.config;
//
//
//import org.apache.hc.client5.http.config.RequestConfig;
//import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
//import org.apache.hc.client5.http.impl.classic.HttpClients;
//import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager;
//import org.apache.hc.core5.util.Timeout;
//import org.springframework.context.annotation.Bean;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
//import org.springframework.web.client.RestTemplate;
//
//import java.time.Duration;
//
//@Configuration
//public class HttpClientConfig {
//
//    @Bean
//    public RestTemplate restTemplate() {
//
//        // Connection pool manager (HttpClient 5.x)
//        PoolingHttpClientConnectionManager connManager = new PoolingHttpClientConnectionManager();
//        connManager.setMaxTotal(200);               // total pooled connections
//        connManager.setDefaultMaxPerRoute(50);      // per-route
//
//        // RequestConfig for HttpClient5 uses org.apache.hc.core5.util.Timeout
//        RequestConfig requestConfig = RequestConfig.custom()
//                .setConnectTimeout(Timeout.ofSeconds(30))
//                .setResponseTimeout(Timeout.ofSeconds(60))   // read timeout
//                .setConnectionRequestTimeout(Timeout.ofSeconds(30))
//                .build();
//
//        CloseableHttpClient client = HttpClients.custom()
//                .setConnectionManager(connManager)
//                .setDefaultRequestConfig(requestConfig)
//                .evictExpiredConnections()
//                .evictIdleConnections(Timeout.ofSeconds(30))
//                .build();
//
//        // Spring's factory accepts CloseableHttpClient (Spring 6)
//        HttpComponentsClientHttpRequestFactory factory = new HttpComponentsClientHttpRequestFactory(client);
//
//        // Optionally set Spring-level timeouts (if available / compatible with your Spring version)
//        // In Spring 6 these setters accept java.time.Duration:
//        try {
//            factory.setConnectTimeout(Duration.ofSeconds(30));
//        } catch (NoSuchMethodError | NoClassDefFoundError e) {
//            // older method signatures may differ; RequestConfig is already doing the heavy lifting
//        }
//
//        return new RestTemplate(factory);
//    }
//}
