import React from "react";

const HistoricalPlaces: React.FC = () => {
  const imgUrls = [
    "https://i.postimg.cc/wBCWYSDJ/fa-6.jpg",
    "https://i.postimg.cc/cCwGNRK1/fa-2.jpg",
    "https://i.postimg.cc/bNr1Lmtj/fa-3.jpg",
    "https://i.postimg.cc/W3ZHD2sX/fa-4.jpg",
    "https://i.postimg.cc/nz1qS52n/fa-5.jpg",
    "https://i.postimg.cc/wBCWYSDJ/fa-6.jpg",
  ];
  return (
    <section className="mt-2" id="historical_nav">
      <div className="historical-places bg-f7fafd pt-32 pb-36">
        <div className="container mx-auto">
          <div className="flex justify-center">
            <div className="col-lg-6">
              <div className="section-title text-center mb-70">
                <div className="text-[20px] font-semibold">
                  HISTORICAL PLACES
                </div>
                <p className="text-center">
                  Lorem ipsum dolor sit amet, consectetur adipisicing elit.
                  Omnis laboriosam totam animi nesciunt cum voluptates.
                </p>
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-8 justify-center">
            {imgUrls.map((img, index) => (
              <div key={index} className="col-lg-4 col-md-6 mb-8">
                <div className="bg-white rounded overflow-hidden transition-all duration-300 shadow-md hover:shadow-lg">
                  <div className="thumb relative overflow-hidden">
                    <img
                      src={img}
                      className="w-[300px] h-[200px] object-cover"
                      alt={`Historical Place ${index + 1}`}
                    />
                    <a
                      href="#"
                      className="absolute top-4 left-4 bg-02acf0 px-6 py-3 inline-block rounded text-white font-bold text-base transition-all duration-300"
                    >
                      $300
                    </a>
                  </div>
                  <div className="dest-info p-5">
                    <a href="#">
                      <h3 className="text-2xl font-bold text-040e27">Lorem</h3>
                    </a>
                    <p className="text-sm font-light text-black mt-2 mb-5">
                      Lorem ipsum dolor sit amet.
                    </p>
                    <div className="rating flex justify-between">
                      <span className="flex items-center">
                        <i className="fas fa-star text-f88408"></i>
                        <i className="fas fa-star text-f88408"></i>
                        <i className="fas fa-star text-f88408"></i>
                        <i className="fas fa-star text-f88408"></i>
                        <i className="fas fa-star text-f88408"></i>
                        <a href="#" className="text-black ml-2">
                          (10 Review)
                        </a>
                      </span>
                      <div className="days flex items-center">
                        <i className="fas fa-clock-o mr-2"></i>
                        <a href="#" className="text-black">
                          2 Days
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="flex justify-center mt-10">
            <div>
              <a href="#" className="btn btn-hist">
                More Places
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HistoricalPlaces;
