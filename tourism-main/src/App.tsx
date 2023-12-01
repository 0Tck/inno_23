import React from "react";
import "./App.css";
import Navbar from "./pages/Navbar";
import Interface from "./pages/Interface";
import HeroSection from "./pages/HeroSection";
import SearchSection from "./pages/SearchSection";

function App() {
  return (
    <div>
      <Navbar />
      {/* <HeroSection /> */}
      {/* <SearchSection /> */}
      <div className="p-10">
        <Interface />
      </div>
    </div>
  );
}

export default App;
