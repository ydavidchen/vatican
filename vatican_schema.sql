CREATE TABLE vatican_users (
id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
name VARCHAR(30) NOT NULL,
email VARCHAR(50),
password VARCHAR(80),
ark_hash VARCHAR(80) 
);

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('John', 'john@example.com', MD5('Doe'), 'aksdjbdssd68dsisdewoweew66734hsdjyaksdjbdssd68dsisdewoweew66734hsdjy');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Derrick', 'derrick@example.com', MD5('yeah'), 'a89QbjbuUBd68dsisdewoweewrwo8rour4jna89QbjbuUBd68dsisdewoweewrwo8rour4jn');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Won', 'won@example.com', MD5('admin'), '3T09Okxsd68dsisdewoweew&u3h3be3T09Okxsd68dsisdewoweewFF');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('David', 'david@example.com', MD5('Dokjdf'), '83Ge02d68dsisdewoweewPuyrkerK83Ge02d68dsisdewoweewPuyrkerK');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Janice', 'janice@example.com', MD5('abc123'), 'GG783k23haksdjbdssd68dsisdewoweew847HHs83Ge02d68dsisdewoweewPuyrkerK');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Jesse', 'jesse@example.com', MD5('ljdf'), '0IUiudjwoaksdjbdssd68dsisdewoweew67834GH83Ge02d68dsisdewoweewPuyrkerK');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Sarah', 'sarah@example.com', MD5('ldg'), 'Thsd64o32o034iaksdjbdssd68dsisdewoweew83Ge02d68dsisdewoweewPuyrkerK');

INSERT INTO vatican_users (name, email, password, ark_hash)
VALUES ('Robert', 'robert@example.com', MD5('efldfe'), '94kjGHgeaksdjbdssd68dsisdewoweew83Ge02d68dsisdewoweewPuyrkerK');