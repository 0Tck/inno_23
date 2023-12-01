import React from 'react';

const TeamSection: React.FC = () => {
  return (
    <section className="team section-bg mt-5">
      <div className="container">
        <div className="section-title">
          <h2 className="text-center">Team</h2>
          <p className="text-center">
            Lorem ipsum dolor sit amet consectetur adipisicing elit. Corrupti eius, maiores reiciendis suscipit
            aspernatur delectus.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {teamMembers.map((member, index) => (
            <TeamMember key={index} member={member} />
          ))}
        </div>
      </div>
    </section>
  );
};

interface TeamMemberProps {
  member: {
    imgSrc: string;
    socialLinks: {
      facebook: string;
      twitter: string;
      instagram: string;
      linkedin: string;
    };
    name: string;
    position: string;
  };
}

const TeamMember: React.FC<TeamMemberProps> = ({ member }) => {
  return (
    <div className="col-md-6 d-flex align-items-stretch">
      <div className="member">
        <div className="member-img">
          <img src={member.imgSrc} className="img-fluid" alt="" />
          <div className="social">
            <a href={member.socialLinks.facebook}>
              <i className="fab fa-facebook"></i>
            </a>
            <a href={member.socialLinks.twitter}>
              <i className="fab fa-twitter"></i>
            </a>
            <a href={member.socialLinks.instagram}>
              <i className="fab fa-instagram"></i>
            </a>
            <a href={member.socialLinks.linkedin}>
              <i className="fab fa-linkedin"></i>
            </a>
          </div>
        </div>
        <div className="member-info">
          <h4>{member.name}</h4>
          <span>{member.position}</span>
        </div>
      </div>
    </div>
  );
};

const teamMembers = [
  {
    imgSrc: 'https://i.postimg.cc/CK51p2wF/team.jpg',
    socialLinks: {
      facebook: '#',
      twitter: '#',
      instagram: '#',
      linkedin: '#',
    },
    name: 'John Doe',
    position: 'Position Specification',
  },
  // Add more team members as needed
];

export default TeamSection;
