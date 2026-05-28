-- SKIN_CANCER_APP - Base de données MySQL
-- TD 8 - Introduction à l'IA - ENSTAB 2025/2026
-- À exécuter dans phpMyAdmin (XAMPP) ou en ligne de commande MySQL

DROP DATABASE IF EXISTS skin_cancer_db;
CREATE DATABASE skin_cancer_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE skin_cancer_db;

-- Table des utilisateurs (authentification)
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(50) NOT NULL
);

-- Table des patients (historique des analyses)
CREATE TABLE patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    result VARCHAR(20) NOT NULL,
    probability FLOAT NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Utilisateur par défaut : admin / 1234
INSERT INTO users (username, password) VALUES ('admin', '1234');
