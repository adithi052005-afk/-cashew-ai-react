function Dashboard({ user, onLogout }) {
  return (
    <div className="dashboard-page">

      <div className="dashboard-header">
        <div>
          <h1>Cashew AI</h1>
          <p>AI-Powered Cashew Disease & Pest Diagnosis</p>
        </div>

        <button className="logout-button" onClick={onLogout}>
          Logout
        </button>
      </div>

      <div className="dashboard-content">

        <h2>Welcome to Cashew AI 🌱</h2>

        <p>
          You are successfully logged in.
        </p>

        <div className="user-card">
          <h3>Account</h3>
          <p>
            <strong>Email:</strong> {user?.email}
          </p>
        </div>

        <div className="diagnosis-card">
          <h3>Cashew Pest & Disease Diagnosis</h3>

          <p>
            Upload a cashew leaf or plant image to identify
            possible pests and diseases using our AI model.
          </p>

          <button className="diagnosis-button">
            Start Diagnosis
          </button>
        </div>

      </div>

    </div>
  );
}

export default Dashboard;