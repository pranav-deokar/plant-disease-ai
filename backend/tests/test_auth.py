"""
Auth & User API Tests
──────────────────────
Tests registration, login, token refresh, profile management, and API keys.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database_models import User


class TestAuthRegister:
    @pytest.mark.asyncio
    async def test_register_new_user(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/register", json={
            "email": "newfarmer@test.com",
            "username": "newfarmer01",
            "password": "securepass123",
            "full_name": "New Farmer",
            "primary_crops": ["tomato", "potato"],
        })
        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newfarmer@test.com"
        assert data["role"] == "farmer"
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, test_user: User):
        response = await client.post("/api/v1/auth/register", json={
            "email": test_user.email,
            "username": "anothername",
            "password": "securepass123",
        })
        assert response.status_code == 409
        assert "already in use" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_register_weak_password(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/register", json={
            "email": "weak@test.com",
            "username": "weakuser",
            "password": "abc",   # too short
        })
        assert response.status_code == 422


class TestAuthLogin:
    @pytest.mark.asyncio
    async def test_login_valid_credentials(self, client: AsyncClient, test_user: User):
        response = await client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "testpassword123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient, test_user: User):
        response = await client.post("/api/v1/auth/login", json={
            "email": test_user.email,
            "password": "wrongpassword",
        })
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_login_nonexistent_email(self, client: AsyncClient):
        response = await client.post("/api/v1/auth/login", json={
            "email": "nobody@nowhere.com",
            "password": "anypassword",
        })
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_me(self, client: AsyncClient, auth_headers: dict, test_user: User):
        response = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == test_user.email
        assert data["role"] == "farmer"

    @pytest.mark.asyncio
    async def test_get_me_unauthenticated(self, client: AsyncClient):
        response = await client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestUserProfile:
    @pytest.mark.asyncio
    async def test_update_profile(self, client: AsyncClient, auth_headers: dict):
        response = await client.patch("/api/v1/users/me", json={
            "full_name": "Updated Farmer Name",
            "location": "Maharashtra, India",
            "farm_size_ha": 5.5,
        }, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Farmer Name"
        assert data["location"] == "Maharashtra, India"

    @pytest.mark.asyncio
    async def test_get_user_stats(self, client: AsyncClient, auth_headers: dict):
        response = await client.get("/api/v1/users/me/stats", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "avg_confidence" in data
        assert "top_diseases" in data
        assert "member_since" in data


class TestAPIKeys:
    @pytest.mark.asyncio
    async def test_create_api_key(self, client: AsyncClient, auth_headers: dict):
        response = await client.post("/api/v1/users/me/api-keys", json={
            "name": "Test Integration Key",
            "scopes": ["predict", "read"],
        }, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("pda_")
        assert "warning" in data   # must remind user to store it

    @pytest.mark.asyncio
    async def test_list_api_keys(self, client: AsyncClient, auth_headers: dict):
        # Create one first
        await client.post("/api/v1/users/me/api-keys", json={
            "name": "Key for listing test",
        }, headers=auth_headers)

        response = await client.get("/api/v1/users/me/api-keys", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_revoke_api_key(self, client: AsyncClient, auth_headers: dict):
        # Create
        create_r = await client.post("/api/v1/users/me/api-keys", json={
            "name": "Key to revoke",
        }, headers=auth_headers)
        key_id = create_r.json()["id"]

        # Revoke
        delete_r = await client.delete(f"/api/v1/users/me/api-keys/{key_id}", headers=auth_headers)
        assert delete_r.status_code == 204

        # Should no longer appear in list
        list_r = await client.get("/api/v1/users/me/api-keys", headers=auth_headers)
        key_ids = [k["id"] for k in list_r.json()]
        assert key_id not in key_ids
