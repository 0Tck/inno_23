import React from 'react';

const HeroSection: React.FC = () => {
  return (
    <section className="hero-wrap" style={{ backgroundImage: "url('https://i.postimg.cc/RVYKYw7J/hero.jpg')" }}>
      <div className="container">
        <div className="row no-gutters main-text align-items-center">
          <div className="col-lg-7 col-md-6 d-flex align-items-end">
            <div className="text">
              <h1 className="mb-4 text-capitalize">find a perfect deal<br /> for your next trip</h1>
              <p style={{ fontSize: '18px' }}>
                Lorem ipsum dolor sit amet consectetur adipisicing elit. Voluptate perspiciatis ducimus aliquam architecto possimus. Voluptatum?
              </p>
              <a href="#" className="btn btn-hero py-2 px-3">View all</a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
