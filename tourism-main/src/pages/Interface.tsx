import React from "react";
import { Route, Routes } from "react-router-dom";
import Home from "./Home";
import Login from "./Login";
import SignUp from "./SignUp";
import HistoricalPlaces from "./HistoricalPlaces";
import WelcomeSection from "./WelcomeSection";
import PopularPlaces from "./PopularPlaces";
import ServicesSection from "./Services";
import TeamSection from "./TeamSection";

function Interface() {
  return (
    <div>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/about" element={<WelcomeSection />} />
        <Route path="/historicalplaces" element={<HistoricalPlaces />} />
        <Route path="/popularplaces" element={<PopularPlaces />} />
        <Route path="/servicessection" element={<ServicesSection />} />
        <Route path="/teamsection" element={<TeamSection />} />
      </Routes>
    </div>
  );
}

export default Interface;
