import { useState } from "react";
import "./App.css";
import Dashboard from "./Dashboard";
import Register from "./Register";
import ForgotPassword from "./ForgotPassword";
import ResetPassword from "./ResetPassword";

function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");

  const [loggedInUser, setLoggedInUser] = useState(null);
  const [showRegister, setShowRegister] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);

  // Show Reset Password page when opened from the email link
  if (window.location.pathname === "/reset-password") {
    return (
      <ResetPassword
        onBackToLogin={() => {
          window.history.pushState({}, "", "/");
          window.location.reload();
        }}
      />
    );
  }

  const handleLogin = async (event) => {
    event.preventDefault();

    setMessage("Logging in...");

    try {
      const response = await fetch(
        "http://192.168.0.107:5000/login",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            email: email.trim(),
            password: password,
          }),
        }
      );

      const data = await response.json();

      if (response.ok) {
        setLoggedInUser(data.user);
        setMessage("");
      } else {
        setMessage(data.message || "Login failed.");
      }
    } catch (error) {
      console.error("Login error:", error);
      setMessage("Cannot connect to server.");
    }
  };

  const handleLogout = () => {
    setLoggedInUser(null);
    setEmail("");
    setPassword("");
    setMessage("");
  };

  // Dashboard after successful login
  if (loggedInUser) {
    return (
      <Dashboard
        user={loggedInUser}
        onLogout={handleLogout}
      />
    );
  }

  // Forgot Password page
  if (showForgotPassword) {
    return (
      <ForgotPassword
        onBackToLogin={() => {
          setShowForgotPassword(false);
          setMessage("");
        }}
      />
    );
  }

  // Register page
  if (showRegister) {
    return (
      <Register
        onBackToLogin={() => {
          setShowRegister(false);
          setMessage("");
        }}
      />
    );
  }

  // Login page
  return (
    <div className="login-page">
      <div className="login-card">

        <div className="login-header">
          <h1>Cashew AI</h1>
          <p>
            AI-Powered Cashew Disease & Pest Diagnosis
          </p>
        </div>

        <form onSubmit={handleLogin}>

          <div className="input-group">
            <label>Email</label>

            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(event) =>
                setEmail(event.target.value)
              }
              required
            />
          </div>

          <div className="input-group">
            <label>Password</label>

            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(event) =>
                setPassword(event.target.value)
              }
              required
            />
          </div>

          <div className="login-options">

            <label>
              <input type="checkbox" />
              Remember me
            </label>

            <button
              type="button"
              className="forgot-password"
              onClick={() => {
                setShowForgotPassword(true);
                setMessage("");
              }}
            >
              Forgot Password?
            </button>

          </div>

          <button
            type="submit"
            className="login-button"
          >
            Login
          </button>

          {message && (
            <p className="login-message">
              {message}
            </p>
          )}

        </form>

        <p className="signup-text">
          Don't have an account?{" "}

          <button
            type="button"
            className="signup-button"
            onClick={() => {
              setShowRegister(true);
              setMessage("");
            }}
          >
            Create Account
          </button>
        </p>

      </div>
    </div>
  );
}

export default App;