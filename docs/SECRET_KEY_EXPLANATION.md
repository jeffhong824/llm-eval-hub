# SECRET_KEY 說明

## 什麼是 SECRET_KEY？

`SECRET_KEY` 是一個用於安全目的的密鑰，主要用於：

1. **JWT Token 生成**：用於生成和驗證 JSON Web Tokens
2. **Session 管理**：用於管理用戶會話
3. **數據加密**：用於加密敏感數據
4. **CSRF 保護**：防止跨站請求偽造攻擊

## 如何生成 SECRET_KEY？

### 方法 1：使用 Python

```python
import secrets
import string

# 生成 32 字符的隨機密鑰
secret_key = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))
print(secret_key)
```

### 方法 2：使用 OpenSSL

```bash
# 生成 32 字節的隨機密鑰（Base64 編碼）
openssl rand -base64 32
```

### 方法 3：使用 UUID

```python
import uuid

# 生成 UUID 作為密鑰
secret_key = str(uuid.uuid4()).replace('-', '')
print(secret_key)
```

## 安全建議

1. **長度**：至少 32 字符
2. **隨機性**：使用加密安全的隨機數生成器
3. **保密性**：不要將密鑰提交到版本控制系統
4. **定期更換**：定期更換密鑰（特別是在安全事件後）
5. **環境分離**：開發、測試、生產環境使用不同的密鑰

## 在 LLM Evaluation Hub 中的使用

在我們的平台中，`SECRET_KEY` 用於：

- 生成 API 訪問令牌
- 加密評估結果
- 保護上傳的文件
- 管理用戶會話

## 配置示例

```bash
# .env 文件
SECRET_KEY=your-generated-secret-key-here-32-chars-minimum

# 或者使用更長的密鑰
SECRET_KEY=your-very-long-and-secure-secret-key-that-is-hard-to-guess-and-contains-mixed-characters-123456789
```

## 注意事項

⚠️ **重要**：
- 永遠不要在代碼中硬編碼 SECRET_KEY
- 不要將包含真實 SECRET_KEY 的 .env 文件提交到 Git
- 在生產環境中使用強隨機密鑰
- 定期輪換密鑰以提高安全性

## 故障排除

如果遇到 SECRET_KEY 相關的錯誤：

1. **檢查長度**：確保密鑰至少 32 字符
2. **檢查字符**：避免使用特殊字符（如引號、空格）
3. **檢查環境變量**：確保 .env 文件被正確加載
4. **重新生成**：如果懷疑密鑰洩露，立即重新生成



